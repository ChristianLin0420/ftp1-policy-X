from __future__ import annotations

import json
import pathlib
import sys

import pytest

from scripts import mot_jepa_control_v2_provenance_batch as provenance
from scripts import mot_jepa_control_v2_verify_gpu_coverage as coverage
from scripts.test_control_v2_merge_collection import _write_container
from scripts.test_control_v2_merge_collection import _write_external_assets
from scripts.test_control_v2_merge_collection import _write_profile
from scripts.test_control_v2_merge_collection import _write_shard

_NODE = "gpu-h100-fixture"


def _gpu_uuid(index: int) -> str:
    return f"GPU-{index + 1:08x}-0000-0000-0000-{index + 1:012x}"


def _rewrite_topology(root: pathlib.Path, *, job_id: str, index: int, gpu_uuid: str | None = None) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    topology = {
        "schema_version": 1,
        "job_id": job_id,
        "node": _NODE,
        "shard_rank": manifest["shard_rank"],
        "gpu_uuid": gpu_uuid or _gpu_uuid(index),
        "http_port": 30_000 + index,
    }
    topology_path = root / "GPU_TOPOLOGY.json"
    topology_path.write_text(json.dumps(topology))
    shard = {
        "schema_version": 1,
        "workers": 1,
        "shard_rank": manifest["shard_rank"],
        "shard_count": manifest["shard_count"],
        "global_episodes": manifest["global_episodes"],
        "gpu_topology": topology,
        "http_port_base": topology["http_port"],
        "episodes": [{"fixture": index}],
    }
    shard_path = root / "SHARD_COLLECTION.json"
    shard_path.write_text(json.dumps(shard))
    manifest["job_id"] = job_id
    manifest["gpu_topology_sha256"] = provenance.file_sha256(topology_path)
    manifest["shard_collection_sha256"] = provenance.file_sha256(shard_path)
    manifest_path.write_text(json.dumps(manifest))


def _write_roots(
    tmp_path: pathlib.Path,
    *,
    split_profile_at: int | None = None,
    split_reference_profile: bool = False,
) -> tuple[list[pathlib.Path], pathlib.Path]:
    shared = tmp_path / "shared"
    profile = _write_profile(shared / "repo", marker="common")
    alternate_profile = _write_profile(shared / "alternate_repo", marker="different")
    container = _write_container(shared / "univtac_full.sqsh")
    external_assets = _write_external_assets(shared / "tacex-assets")
    local_assets = shared / "local-assets"
    roots = []
    for index in range(coverage.EXPECTED_GPU_COUNT):
        job_id = str(91_000 + index)
        root = (tmp_path / f"lift_bottle_{job_id}").resolve()
        selected_profile = alternate_profile if split_profile_at == index else profile
        _write_shard(
            root,
            rank=index % 4,
            profile_roots=selected_profile,
            local_assets=local_assets,
            container=container,
            external_assets=external_assets,
        )
        _rewrite_topology(root, job_id=job_id, index=index)
        roots.append(root)
    reference = (tmp_path / "lift_bottle_91999").resolve()
    _write_shard(
        reference,
        rank=0,
        profile_roots=alternate_profile if split_reference_profile else profile,
        local_assets=local_assets,
        container=container,
        external_assets=external_assets,
        shard_count=20,
        episode_quota=50,
        collection_mode="production",
    )
    _rewrite_topology(reference, job_id="91999", index=99)
    return roots, reference


def test_verifies_eight_distinct_physical_gpus_with_full_provenance(tmp_path: pathlib.Path) -> None:
    roots, reference = _write_roots(tmp_path)

    result = coverage.verify_gpu_coverage(roots, reference_root=reference, expected_node=_NODE)

    assert result["status"] == "VERIFIED"
    assert result["expected_count"] == 8
    assert len(set(result["job_ids"])) == 8
    assert len(set(result["gpu_uuids"])) == 8
    assert len(set(result["http_ports"])) == 8
    assert len(result["source_profile_sha256"]) == 64
    assert len(result["input_set_sha256"]) == 64
    assert result["reference"]["root"] == str(reference)
    assert result["reference"]["job_id"] == "91999"
    assert set(result["reference"]["input_sha256"]) == set(coverage.HASHED_INPUTS)
    assert all(set(row["input_sha256"]) == set(coverage.HASHED_INPUTS) for row in result["roots"])


@pytest.mark.parametrize("marker", ["FAILED.json", "HEARTBEAT.json"])
def test_rejects_failure_or_running_marker(tmp_path: pathlib.Path, marker: str) -> None:
    roots, reference = _write_roots(tmp_path)
    (roots[0] / marker).write_text("{}\n")

    with pytest.raises(ValueError, match="failure/running markers"):
        coverage.verify_gpu_coverage(roots, reference_root=reference, expected_node=_NODE)


def test_rejects_duplicate_physical_gpu_uuid(tmp_path: pathlib.Path) -> None:
    roots, reference = _write_roots(tmp_path)
    _rewrite_topology(roots[-1], job_id="91007", index=7, gpu_uuid=_gpu_uuid(0))

    with pytest.raises(ValueError, match="eight distinct physical GPU UUIDs"):
        coverage.verify_gpu_coverage(roots, reference_root=reference, expected_node=_NODE)


def test_rejects_different_shared_source_profile(tmp_path: pathlib.Path) -> None:
    roots, reference = _write_roots(tmp_path, split_profile_at=7)

    with pytest.raises(ValueError, match="source code/profile closure differs from the production reference"):
        coverage.verify_gpu_coverage(roots, reference_root=reference, expected_node=_NODE)


def test_rejects_source_profile_that_differs_from_reference(tmp_path: pathlib.Path) -> None:
    roots, reference = _write_roots(tmp_path, split_reference_profile=True)

    with pytest.raises(ValueError, match="source code/profile closure differs from the production reference"):
        coverage.verify_gpu_coverage(roots, reference_root=reference, expected_node=_NODE)


def test_reference_must_retain_exact_committed_marker(tmp_path: pathlib.Path) -> None:
    roots, reference = _write_roots(tmp_path)
    (reference / "DONE").write_text("1\n")

    with pytest.raises(ValueError, match="reference DONE must contain exactly"):
        coverage.verify_gpu_coverage(roots, reference_root=reference, expected_node=_NODE)


def test_reference_manifest_accepts_absolute_lustre_alias_to_same_in_root_files(tmp_path: pathlib.Path) -> None:
    roots, reference = _write_roots(tmp_path)
    alias = tmp_path / "reference_alias"
    alias.symlink_to(reference, target_is_directory=True)
    manifest_path = reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for field, name in {
        "runtime_config": "mot_control_v2.yml",
        "source_provenance": "source_provenance.json",
        "shard_collection": "SHARD_COLLECTION.json",
        "gpu_topology": "GPU_TOPOLOGY.json",
    }.items():
        manifest[field] = str(alias / name)
    manifest_path.write_text(json.dumps(manifest))

    result = coverage.verify_gpu_coverage(roots, reference_root=reference, expected_node=_NODE)

    assert result["status"] == "VERIFIED"


def test_reference_manifest_rejects_relative_in_root_file(tmp_path: pathlib.Path) -> None:
    roots, reference = _write_roots(tmp_path)
    manifest_path = reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime_config"] = "mot_control_v2.yml"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="runtime_config is not the canonical in-root member"):
        coverage.verify_gpu_coverage(roots, reference_root=reference, expected_node=_NODE)


def test_reference_manifest_rejects_traversing_alias_to_in_root_file(tmp_path: pathlib.Path) -> None:
    roots, reference = _write_roots(tmp_path)
    (reference / "nested").mkdir()
    manifest_path = reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime_config"] = str(reference / "nested" / ".." / "mot_control_v2.yml")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="runtime_config is not the canonical in-root member"):
        coverage.verify_gpu_coverage(roots, reference_root=reference, expected_node=_NODE)


def test_rejects_noncanonical_root_and_nonexact_done(tmp_path: pathlib.Path) -> None:
    roots, reference = _write_roots(tmp_path)
    alias = tmp_path / "lift_bottle_91000_alias"
    alias.symlink_to(roots[0], target_is_directory=True)
    with pytest.raises(ValueError, match="canonical and non-symlinked"):
        coverage.verify_gpu_coverage([alias, *roots[1:]], reference_root=reference, expected_node=_NODE)

    (roots[0] / "DONE").write_text("1")
    with pytest.raises(ValueError, match="DONE must contain exactly"):
        coverage.verify_gpu_coverage(roots, reference_root=reference, expected_node=_NODE)


def test_explicit_job_ids_are_positional_and_atomic_output_is_written(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "nested/coverage.json"
    expected = {"schema_version": 1, "status": "VERIFIED"}
    monkeypatch.setattr(coverage, "verify_gpu_coverage", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify",
            "--root",
            "/fixture/root",
            "--reference-root",
            "/fixture/reference",
            "--expected-node",
            _NODE,
            "--expected-job",
            "91000",
            "--output",
            str(output),
        ],
    )

    coverage.main()

    assert json.loads(output.read_text()) == expected
    assert output.read_bytes().endswith(b"\n")
