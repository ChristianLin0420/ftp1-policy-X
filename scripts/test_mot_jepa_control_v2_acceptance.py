from __future__ import annotations

import hashlib
import json
import types

import pytest

from scripts import mot_jepa_control_v2_acceptance as acceptance


def _write_qualified_runtime(tmp_path, *, official: bool):
    payload = {
        "schema_version": 1,
        "qualification": (
            acceptance.OFFICIAL_RUNTIME_QUALIFICATION if official else acceptance.COMMON_RUNTIME_QUALIFICATION
        ),
        "common": {"container": {}, "external_assets": {}, "local_assets": {}},
    }
    if official:
        payload["official"] = {"checkpoint": {}, "overlay": {}, "ready": {}, "tokenizer": {}}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["qualification_sha256"] = hashlib.sha256(canonical).hexdigest()
    path = tmp_path / ("official_runtime.json" if official else "common_runtime.json")
    path.write_text(json.dumps(payload))
    return path


def _write_run(tmp_path, *, start_seed: int = 17, success: int = 99):
    root = tmp_path / "run"
    (root / "metadata").mkdir(parents=True)
    entries = {
        str(seed): {"result": "success" if offset < success else "failed"}
        for offset, seed in enumerate(range(start_seed, start_seed + 100))
    }
    (root / "metadata" / "worker_0.json").write_text(json.dumps(entries))
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "total_episodes": 100,
                "success": success,
                "failed": 100 - success,
                "error": 0,
                "eval_args": {"start_seed": start_seed, "total_num": 100},
            }
        )
    )
    return root


def test_load_episode_results_reads_exact_seed_records(tmp_path) -> None:
    root = _write_run(tmp_path)
    results, entries = acceptance.load_episode_results(root, start_seed=17)
    assert len(results) == len(entries) == 100
    assert sum(results.values()) == 99
    assert results[17]
    assert not results[116]


def test_result_metadata_manifest_binds_aggregate_and_worker_records(tmp_path) -> None:
    root = _write_run(tmp_path)
    before = acceptance.result_metadata_manifest(root)

    worker = root / "metadata/worker_0.json"
    payload = json.loads(worker.read_text())
    payload["17"]["diagnostic"] = "changed"
    worker.write_text(json.dumps(payload))
    after = acceptance.result_metadata_manifest(root)

    assert set(before["files"]) == {"metadata.json", "metadata/worker_0.json"}
    assert before["content_sha256"] != after["content_sha256"]


def test_load_episode_results_rejects_missing_seed_even_if_summary_says_n100(tmp_path) -> None:
    root = _write_run(tmp_path)
    path = root / "metadata" / "worker_0.json"
    entries = json.loads(path.read_text())
    entries.pop("42")
    path.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="seed set differs"):
        acceptance.load_episode_results(root, start_seed=17)


def test_load_episode_results_rejects_aggregate_disagreement(tmp_path) -> None:
    root = _write_run(tmp_path)
    summary_path = root / "metadata.json"
    summary = json.loads(summary_path.read_text())
    summary["success"] = 100
    summary["failed"] = 0
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="aggregate metadata disagrees"):
        acceptance.load_episode_results(root, start_seed=17)


@pytest.mark.parametrize("official", [False, True])
def test_qualified_runtime_digest_and_mode_close(official, tmp_path) -> None:
    path = _write_qualified_runtime(tmp_path, official=official)

    payload = acceptance.load_qualified_runtime(path, require_official=official)

    assert payload["qualification_sha256"]
    if not official:
        with pytest.raises(ValueError, match="invalid qualified-runtime field set"):
            acceptance.load_qualified_runtime(path, require_official=True)


def test_qualified_runtime_rejects_tampering(tmp_path) -> None:
    path = _write_qualified_runtime(tmp_path, official=True)
    payload = json.loads(path.read_text())
    payload["common"]["container"]["bytes"] = 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="digest does not close"):
        acceptance.load_qualified_runtime(path, require_official=True)


def test_deployment_contract_allows_step_zero_validation_selection(tmp_path, monkeypatch) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text("{}")
    normalization_path = tmp_path / "normalization.json"
    normalization_path.write_text("{}")
    checkpoint = tmp_path / "checkpoints" / "best_validation_0"
    checkpoint.mkdir(parents=True)
    identities = [
        (
            f"/data/control.zarr\0episode=7\0start={12 + index}\0stride=2"
            f"\0dataset_index={index}\0time_bin={('early', 'middle', 'late')[index % 3]}"
            f"\0phase={index % 4}\0contact={index % 2}\0history_length={index // 2 + 1}"
            f"\0oldest_command_present={index % 2}"
        )
        for index in range(32)
    ]
    sample_sha = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
    (tmp_path / "VALIDATION_SAMPLES.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sampling": "all_episode_time_phase_contact_cold_start_stratified_v2",
                "requested_examples": 32,
                "sample_count": 32,
                "episode_count": 1,
                "sample_sha256": sample_sha,
                "episodes": [{"identity": "/data/control.zarr\0episode=7", "sample_count": 32}],
                "samples": [
                    {"dataset_index": index, "identity": identity} for index, identity in enumerate(identities)
                ],
            }
        )
    )
    (tmp_path / "BEST_VALIDATION.json").write_text(
        json.dumps(
            {
                "schema_version": acceptance.CONTROL_V2_ARTIFACT_SCHEMA,
                "selection_metric": "validation_action_loss",
                "selection_constraint": "validation_safety_violation_values==0",
                "selection_mode": "min",
                "tie_break": "earliest_step",
                "step": 0,
                "validation_action_loss": 0.25,
                "checkpoint": str(checkpoint.resolve()),
                "artifact_sha256": acceptance._sha256(artifact_path),  # noqa: SLF001
                "normalization_sha256": acceptance._sha256(normalization_path),  # noqa: SLF001
                "source_store_sha256": "c" * 64,
                "validation_action_mae": 0.2,
                "validation_phase_loss": 0.3,
                "validation_contact_loss": 0.4,
                "validation_contact_count": 16.0,
                "validation_clean_order_loss": 0.5,
                "validation_overlap_loss": 0.1,
                "validation_safety_loss": 0.01,
                "validation_safety_tube_loss": 0.02,
                "validation_safety_tube_mean_loss": 0.005,
                "validation_safety_tube_tail_loss": 0.015,
                "validation_safety_tube_max_ratio": 0.8,
                "validation_safety_tube_min_slack": 0.01,
                "validation_joint_violation_values": 0.0,
                "validation_rate_violation_values": 0.0,
                "validation_safety_violation_values": 0.0,
                "validation_joint_max_excess_ratio": 0.0,
                "validation_rate_max_ratio": 0.95,
                "validation_examples": 32.0,
                "validation_episode_count": 1.0,
                "validation_objective": 0.6,
                "validation_sampling": "all_episode_time_phase_contact_cold_start_stratified_v2",
                "validation_sample_sha256": sample_sha,
            }
        )
    )
    production_checks = []
    sentinel = types.SimpleNamespace(
        source_store_sha256="c" * 64,
        require_production_data=lambda: production_checks.append(True),
    )
    monkeypatch.setattr(
        acceptance.ControlV2Artifact,
        "from_json",
        classmethod(lambda _cls, _text: sentinel),
    )

    artifact, selection, _ = acceptance.load_deployment_contract(artifact_path)

    assert artifact is sentinel
    assert production_checks == [True]
    assert selection["step"] == 0

    original = json.loads((tmp_path / "BEST_VALIDATION.json").read_text())
    for field, value, message in (
        ("selection_constraint", "none", "selection_constraint"),
        ("validation_joint_violation_values", 1.0, "zero held-out safety constraint"),
        ("validation_rate_violation_values", 1.0, "zero held-out safety constraint"),
        ("validation_safety_violation_values", 1.0, "zero held-out safety constraint"),
        ("validation_joint_max_excess_ratio", 0.01, "joint_max_excess_ratio must be zero"),
        ("validation_rate_max_ratio", 1.0001, "rate_max_ratio must be at most 1"),
        ("artifact_sha256", "a" * 64, "artifact_sha256 does not close"),
        ("normalization_sha256", "b" * 64, "normalization_sha256 does not close"),
        ("source_store_sha256", "d" * 64, "source_store_sha256 does not close"),
    ):
        unsafe = {**original, field: value}
        (tmp_path / "BEST_VALIDATION.json").write_text(json.dumps(unsafe))
        with pytest.raises(ValueError, match=message):
            acceptance.load_deployment_contract(artifact_path)

    (tmp_path / "BEST_VALIDATION.json").write_text(json.dumps(original))
    original_manifest = json.loads((tmp_path / "VALIDATION_SAMPLES.json").read_text())
    for old, new in (
        ("\0time_bin=late", "\0time_bin=early"),
        ("\0phase=3", "\0phase=0"),
        ("\0contact=1", "\0contact=0"),
    ):
        manifest = json.loads(json.dumps(original_manifest))
        for sample in manifest["samples"]:
            sample["identity"] = sample["identity"].replace(old, new)
        changed_identities = [sample["identity"] for sample in manifest["samples"]]
        manifest["sample_sha256"] = hashlib.sha256(
            json.dumps(changed_identities, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        (tmp_path / "VALIDATION_SAMPLES.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="required validation strata"):
            acceptance.load_deployment_contract(artifact_path)
