from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from scripts import mot_jepa_control_v2_merge_paired_eval as paired_merge
from scripts import mot_jepa_control_v2_provenance as provenance
from scripts import mot_jepa_control_v2_qualified_runtime as runtime_qualifier

_PROFILE_DIRECTORIES = {
    "UniVTAC/scripts",
    "UniVTAC/envs",
    "UniVTAC/encoder",
    "UniVTAC/policy",
    "UniVTAC/assets",
    "UniVTAC/third_party/TacEx/source/tacex/tacex",
    "UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/sensors",
    "UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/robots",
    "UniVTAC/third_party/TacEx/source/tacex_tasks/tacex_tasks",
    "UniVTAC/third_party/TacEx/source/tacex_uipc/tacex_uipc",
    "src/openpi",
}


def _write_envelope(root: pathlib.Path, start_seed: int, *, contract: str = "c" * 64) -> pathlib.Path:
    root.mkdir()
    for name in ("provenance.json", "qualified_runtime_after.json", "evaluation_contract.json"):
        (root / name).write_text("{}\n")
    student_root = root / "student/lift_bottle/student"
    official_root = root / "official/lift_bottle/official"
    student_root.mkdir(parents=True)
    official_root.mkdir(parents=True)
    seeds = range(start_seed, start_seed + paired_merge.SHARD_TRIALS)
    results = {str(seed): seed % 3 != 0 for seed in seeds}
    traces = {
        str(seed): {"path": str(student_root / f"{seed}.npz"), "sha256": f"{seed:064x}"[-64:], "passed": True}
        for seed in seeds
    }
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "root": str(root.resolve()),
        "start_seed": start_seed,
        "end_seed": start_seed + paired_merge.SHARD_TRIALS - 1,
        "trials": paired_merge.SHARD_TRIALS,
        "artifact": "/qualified/run/artifact.json",
        "artifact_sha256": "a" * 64,
        "selected_step": 10_000,
        "completed_step": 10_000,
        "selection_value": 0.01,
        "qualified_runtime": str((root / "qualified_runtime_after.json").resolve()),
        "qualified_runtime_sha256": "q" * 64,
        "runtime_qualification_sha256": "r" * 64,
        "provenance": str((root / "provenance.json").resolve()),
        "provenance_sha256": "p" * 64,
        "source_content_sha256": "s" * 64,
        "static_source_sha256": "t" * 64,
        "external_runtime_roots": {
            "checkpoint": "/runtime/checkpoint",
            "container": "/runtime/container.sqsh",
            "external_assets": "/runtime/assets",
            "overlay": "/runtime/overlay",
            "tokenizer": "/runtime/tokenizer.model",
        },
        "external_runtime_roots_sha256": "u" * 64,
        "append_manifest": "/submission/paired.json",
        "append_manifest_sha256": "v" * 64,
        "append_launcher": "/repo/control_v2_submit_paired_n100.sh",
        "append_launcher_sha256": "w" * 64,
        "append_submission_sha256": "x" * 64,
        "n20_job_id": "6704613",
        "evaluation_contract": str((root / "evaluation_contract.json").resolve()),
        "evaluation_contract_sha256": "e" * 64,
        "student_root": str(student_root.resolve()),
        "official_root": str(official_root.resolve()),
        "student_metadata": {"content_sha256": f"student-{start_seed}"},
        "official_metadata": {"content_sha256": f"official-{start_seed}"},
        "student_eval_args": {"policy_name": "mot_control_v2", "policy_sample_seed": 0},
        "official_eval_args": {"policy_name": "official_ftp1", "policy_sample_seed": 0},
        "student_seed_results": results,
        "official_seed_results": results,
        "student_trace_manifest": traces,
        "student_trace_content_sha256": paired_merge._canonical_digest(traces),  # noqa: SLF001
        "model_contract_sha256": "m" * 64,
        "contract_sha256": contract,
    }
    assert set(manifest) == paired_merge.SHARD_MANIFEST_FIELDS
    manifest_path = root / "PAIRED_SHARD.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _rewrite_done(root)
    return root


def _rewrite_done(root: pathlib.Path) -> None:
    manifest = json.loads((root / "PAIRED_SHARD.json").read_text())
    marker = {
        "schema_version": 1,
        "status": "COMPLETE",
        "start_seed": manifest["start_seed"],
        "trials": paired_merge.SHARD_TRIALS,
        "manifest_sha256": paired_merge._sha256(root / "PAIRED_SHARD.json"),  # noqa: SLF001
    }
    (root / "DONE").write_text(json.dumps(marker, sort_keys=True) + "\n")


def _write_source_profile(
    tmp_path: pathlib.Path,
    *,
    omit: str | None = None,
) -> tuple[pathlib.Path, pathlib.Path, dict, pathlib.Path, pathlib.Path]:
    repo = tmp_path / "repo"
    paths = []
    for relative in paired_merge.REQUIRED_STATIC_PROFILE_ROOTS:
        path = repo / relative
        if relative in _PROFILE_DIRECTORIES:
            path.mkdir(parents=True, exist_ok=True)
            (path / "fixture.txt").write_text(relative)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative)
        if relative != omit:
            paths.append(path)
    append_launcher = repo / "scripts_exp_zarr/mot_jepa/control_v2_submit_paired_n100.sh"
    append_manifest = tmp_path / "append_submission.json"
    append_manifest.write_text("{}")
    if omit != "append_manifest":
        paths.append(append_manifest)
    shard = tmp_path / "shard_source"
    shard.mkdir()
    local = shard / "evaluation_contract.json"
    local.write_text("{}")
    runtime_root = tmp_path / "runtime"
    container = runtime_root / "container.sqsh"
    external_assets = runtime_root / "assets"
    overlay = runtime_root / "overlay"
    checkpoint = runtime_root / "checkpoint"
    tokenizer = runtime_root / "tokenizer.model"
    container.parent.mkdir(parents=True, exist_ok=True)
    container.write_text("container")
    external_assets.mkdir()
    (external_assets / "asset.bin").write_text("asset")
    overlay.mkdir()
    ready = overlay / "READY.json"
    ready.write_text("ready")
    checkpoint.mkdir()
    checkpoint_records = {}
    for relative, (expected_bytes, expected_sha256) in runtime_qualifier.FTP1_FILES.items():
        checkpoint_member = checkpoint / relative
        checkpoint_member.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_member.write_text(relative)
        checkpoint_records[relative] = {
            "path": str(checkpoint_member.resolve()),
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        }
    tokenizer.write_text("tokenizer")
    external = {
        "container": container,
        "external_assets": external_assets,
        "overlay": overlay,
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
    }
    stat_only = [path for name, path in external.items() if omit != f"external:{name}"]
    payload = provenance.snapshot([*paths, local], stat_only_paths=stat_only)
    manifest = shard / "provenance.json"
    manifest.write_text(json.dumps(payload))
    qualified_runtime = {
        "common": {
            "container": {
                "path": str(container.resolve()),
                "bytes": runtime_qualifier.CONTAINER[0],
                "sha256": runtime_qualifier.CONTAINER[1],
            },
            "external_assets": {
                "path": str(external_assets.resolve()),
                "file_count": runtime_qualifier.EXTERNAL_ASSET_TREE[0],
                "bytes": runtime_qualifier.EXTERNAL_ASSET_TREE[1],
                "sha256": runtime_qualifier.EXTERNAL_ASSET_TREE[2],
            },
            "local_assets": {
                "path": str((repo / "UniVTAC").resolve()),
                "file_count": runtime_qualifier.LOCAL_ASSET_TREE[0],
                "bytes": runtime_qualifier.LOCAL_ASSET_TREE[1],
                "sha256": runtime_qualifier.LOCAL_ASSET_TREE[2],
            },
        },
        "official": {
            "checkpoint": checkpoint_records,
            "overlay": {
                "path": str(overlay.resolve()),
                "file_count": runtime_qualifier.FTP1_OVERLAY_TREE[0],
                "bytes": runtime_qualifier.FTP1_OVERLAY_TREE[1],
                "sha256": runtime_qualifier.FTP1_OVERLAY_TREE[2],
            },
            "ready": {
                "path": str(ready.resolve()),
                "bytes": runtime_qualifier.READY[0],
                "sha256": runtime_qualifier.READY[1],
            },
            "tokenizer": {
                "path": str(tokenizer.resolve()),
                "bytes": runtime_qualifier.TOKENIZER[0],
                "sha256": runtime_qualifier.TOKENIZER[1],
            },
        },
    }
    return manifest, shard, qualified_runtime, append_manifest.resolve(), append_launcher.resolve()


@pytest.fixture
def shard_roots(tmp_path: pathlib.Path) -> list[pathlib.Path]:
    return [
        _write_envelope(tmp_path / f"shard_{index}", start) for index, start in enumerate(paired_merge.SHARD_STARTS)
    ]


def test_exact_five_contiguous_shard_envelopes_close_n100(shard_roots) -> None:
    envelopes = paired_merge.load_shard_envelopes(list(reversed(shard_roots)))

    assert [manifest["start_seed"] for _, manifest in envelopes] == list(paired_merge.SHARD_STARTS)
    assert sum(len(manifest["student_seed_results"]) for _, manifest in envelopes) == 100
    assert paired_merge.FIRST_GRIPPER_JUMP_MAX == 0.005


def test_missing_seed_is_rejected(shard_roots) -> None:
    path = shard_roots[2] / "PAIRED_SHARD.json"
    manifest = json.loads(path.read_text())
    manifest["student_seed_results"].pop(str(manifest["start_seed"] + 7))
    path.write_text(json.dumps(manifest))
    _rewrite_done(shard_roots[2])

    with pytest.raises(ValueError, match="seed set differs: missing"):
        paired_merge.load_shard_envelopes(shard_roots)


def test_duplicate_seed_across_shards_is_rejected(shard_roots) -> None:
    path = shard_roots[1] / "PAIRED_SHARD.json"
    manifest = json.loads(path.read_text())
    manifest["student_seed_results"].pop(str(manifest["start_seed"] + 1))
    manifest["student_seed_results"][str(paired_merge.SHARD_STARTS[0])] = True
    path.write_text(json.dumps(manifest))
    _rewrite_done(shard_roots[1])

    with pytest.raises(ValueError, match="duplicate student seeds"):
        paired_merge.load_shard_envelopes(shard_roots)


def test_manifest_tampering_breaks_done_digest(shard_roots) -> None:
    path = shard_roots[0] / "PAIRED_SHARD.json"
    manifest = json.loads(path.read_text())
    manifest["selection_value"] = 0.02
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="DONE/manifest digest differs"):
        paired_merge.load_shard_envelopes(shard_roots)


@pytest.mark.parametrize("field", ["static_source_sha256", "qualified_runtime_sha256", "contract_sha256"])
def test_cross_shard_contract_tampering_is_rejected(shard_roots, field) -> None:
    path = shard_roots[-1] / "PAIRED_SHARD.json"
    manifest = json.loads(path.read_text())
    manifest[field] = "x" * 64
    path.write_text(json.dumps(manifest))
    _rewrite_done(shard_roots[-1])

    with pytest.raises(ValueError, match=f"contracts differ for {field}"):
        paired_merge.load_shard_envelopes(shard_roots)


def test_requires_exactly_five_distinct_roots(shard_roots) -> None:
    with pytest.raises(ValueError, match="exactly five distinct"):
        paired_merge.load_shard_envelopes(shard_roots[:4])
    with pytest.raises(ValueError, match="exactly five distinct"):
        paired_merge.load_shard_envelopes([*shard_roots[:4], shard_roots[0]])


def test_static_source_contract_closes_full_versioned_profile(tmp_path) -> None:
    manifest, shard, runtime, append_manifest, append_launcher = _write_source_profile(tmp_path)

    digest, payload, repo, external = paired_merge._static_source_contract(  # noqa: SLF001
        manifest,
        shard.resolve(),
        runtime,
        append_manifest,
        append_launcher,
    )

    assert len(digest) == 64
    assert payload["file_count"] >= len(paired_merge.REQUIRED_STATIC_PROFILE_ROOTS)
    assert repo.name == "repo"
    assert set(external) == set(paired_merge.EXTERNAL_RUNTIME_ROOT_NAMES)


@pytest.mark.parametrize("omitted", paired_merge.REQUIRED_STATIC_PROFILE_ROOTS)
def test_static_source_contract_rejects_every_omitted_required_root(tmp_path, omitted) -> None:
    manifest, shard, runtime, append_manifest, append_launcher = _write_source_profile(tmp_path, omit=omitted)

    with pytest.raises(ValueError, match="source closure lacks"):
        paired_merge._static_source_contract(  # noqa: SLF001
            manifest, shard.resolve(), runtime, append_manifest, append_launcher
        )


def test_static_source_contract_rejects_omitted_append_manifest(tmp_path) -> None:
    manifest, shard, runtime, append_manifest, append_launcher = _write_source_profile(tmp_path, omit="append_manifest")

    with pytest.raises(ValueError, match="hashed append manifest root"):
        paired_merge._static_source_contract(  # noqa: SLF001
            manifest, shard.resolve(), runtime, append_manifest, append_launcher
        )


@pytest.mark.parametrize("omitted", paired_merge.EXTERNAL_RUNTIME_ROOT_NAMES)
def test_static_source_contract_rejects_every_omitted_external_root(tmp_path, omitted) -> None:
    manifest, shard, runtime, append_manifest, append_launcher = _write_source_profile(
        tmp_path, omit=f"external:{omitted}"
    )

    with pytest.raises(ValueError, match=f"stat-only {omitted} root"):
        paired_merge._static_source_contract(  # noqa: SLF001
            manifest, shard.resolve(), runtime, append_manifest, append_launcher
        )


@pytest.mark.parametrize("changed", paired_merge.EXTERNAL_RUNTIME_ROOT_NAMES)
def test_static_source_contract_rejects_external_root_marked_content_hashed(tmp_path, changed) -> None:
    manifest, shard, runtime, append_manifest, append_launcher = _write_source_profile(tmp_path)
    payload = json.loads(manifest.read_text())
    changed_path = runtime["common"].get(changed, runtime["official"].get(changed))
    if changed == "checkpoint":
        changed_path = {
            "path": str(pathlib.Path(runtime["official"]["checkpoint"]["model.safetensors"]["path"]).parent)
        }
    target = changed_path["path"]
    for record in payload["roots"]:
        if record["path"] == target:
            record["content_hashed"] = True
            break
    else:
        raise AssertionError(target)
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=f"stat-only {changed} root"):
        paired_merge._static_source_contract(  # noqa: SLF001
            manifest, shard.resolve(), runtime, append_manifest, append_launcher
        )


def test_runtime_contract_rejects_missing_or_extra_checkpoint_members(tmp_path) -> None:
    _manifest, _shard, runtime, _append_manifest, _append_launcher = _write_source_profile(tmp_path)
    repo = (tmp_path / "repo").resolve()
    missing = json.loads(json.dumps(runtime))
    missing["official"]["checkpoint"].pop("model.safetensors")
    with pytest.raises(ValueError, match="exact official FTP1 checkpoint members"):
        paired_merge._runtime_external_roots(missing, repo)  # noqa: SLF001

    extra = json.loads(json.dumps(runtime))
    extra["official"]["checkpoint"]["unexpected.bin"] = extra["official"]["checkpoint"]["metadata.pt"]
    with pytest.raises(ValueError, match="exact official FTP1 checkpoint members"):
        paired_merge._runtime_external_roots(extra, repo)  # noqa: SLF001


def test_static_source_contract_rejects_an_extra_stat_only_root(tmp_path) -> None:
    manifest, shard, runtime, append_manifest, append_launcher = _write_source_profile(tmp_path)
    extra = tmp_path / "unexpected_external.bin"
    extra.write_text("unexpected")
    payload = json.loads(manifest.read_text())
    extra_payload = provenance.snapshot([], stat_only_paths=[extra])
    payload["roots"].extend(extra_payload["roots"])
    payload["files"].update(extra_payload["files"])
    payload["file_count"] = len(payload["files"])
    index = "".join(
        f"{path}\0{record['bytes']}\0{record['mtime_ns']}\0{record['inode']}\0{record['sha256']}\n"
        for path, record in sorted(payload["files"].items())
    )
    payload["content_sha256"] = hashlib.sha256(index.encode()).hexdigest()
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unexpected stat-only root set"):
        paired_merge._static_source_contract(  # noqa: SLF001
            manifest, shard.resolve(), runtime, append_manifest, append_launcher
        )


def test_evaluation_contract_requires_the_preregistered_node() -> None:
    contract = {
        "schema_version": 2,
        "job_id": "123",
        "requested_node": paired_merge.EVAL_NODE,
        "actual_node_list": paired_merge.EVAL_NODE,
        "mode": "paired_shard",
        "trials": 20,
        "start_seed": paired_merge.SHARD_STARTS[0],
        "selected_step": 10_000,
        "completed_step": 10_000,
        "append_manifest": "/submission/paired.json",
        "append_launcher": "/repo/control_v2_submit_paired_n100.sh",
        "created_unix": 1.0,
    }
    paired_merge._validate_evaluation_contract(  # noqa: SLF001
        contract,
        start_seed=paired_merge.SHARD_STARTS[0],
        selected_step=10_000,
        completed_step=10_000,
    )

    for field in ("requested_node", "actual_node_list"):
        tampered = dict(contract)
        tampered[field] = "gpu-h100-0001"
        with pytest.raises(ValueError, match="evaluation contract differs"):
            paired_merge._validate_evaluation_contract(  # noqa: SLF001
                tampered,
                start_seed=paired_merge.SHARD_STARTS[0],
                selected_step=10_000,
                completed_step=10_000,
            )


def _append_submission_fixture(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, dict, dict]:
    repo = tmp_path / "repo"
    launcher = repo / "scripts_exp_zarr/mot_jepa/control_v2_submit_paired_n100.sh"
    shard_sbatch = repo / "scripts_exp_zarr/mot_jepa/control_v2_closedloop_paired_shard.sbatch"
    merge_sbatch = repo / "scripts_exp_zarr/mot_jepa/control_v2_merge_paired_eval.sbatch"
    for path in (launcher, shard_sbatch, merge_sbatch):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
    pair_parent = (tmp_path / "paired").resolve()
    jobs = [str(8_100_001 + index) for index in range(5)]
    roots = [
        str(pair_parent / f"lift_bottle_s{seed}_n20_{job}")
        for seed, job in zip(paired_merge.SHARD_STARTS, jobs, strict=True)
    ]
    shard_root = pathlib.Path(roots[0])
    shard_root.mkdir(parents=True)
    artifact = (tmp_path / "v2/artifact.json").resolve()
    artifact.parent.mkdir()
    artifact.write_text("{}")
    merge_job = "8100006"
    payload = {
        "schema_version": 1,
        "status": "SUBMITTED",
        "v2_run": str(artifact.parent),
        "n20_job_id": "6704613",
        "n20_dependency": "afterok:6704613",
        "pair_parent": str(pair_parent),
        "eval_node": paired_merge.EVAL_NODE,
        "shard": {
            "partition": "batch",
            "time_limit": "04:00:00",
            "run_seconds_per_policy": 5400,
            "trials_each": 20,
            "start_seeds": list(paired_merge.SHARD_STARTS),
            "job_ids": jobs,
            "roots": roots,
        },
        "merge": {
            "partition": "cpu",
            "time_limit": "02:00:00",
            "dependency": f"afterok:{':'.join(jobs)}",
            "job_id": merge_job,
            "root": str(pair_parent / f"lift_bottle_s1000000_n100_{merge_job}"),
        },
        "job_ids": [*jobs, merge_job],
        "sources": {
            name: {"path": str(path.resolve()), "sha256": paired_merge._sha256(path)}  # noqa: SLF001
            for name, path in {
                "launcher": launcher,
                "shard_sbatch": shard_sbatch,
                "merge_sbatch": merge_sbatch,
            }.items()
        },
        "submitted_unix": 1.0,
    }
    manifest = (tmp_path / "append.json").resolve()
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    evaluation = {
        "job_id": jobs[0],
        "append_manifest": str(manifest),
        "append_launcher": str(launcher.resolve()),
    }
    return manifest, launcher.resolve(), artifact, evaluation, payload


def test_append_submission_contract_closes_current_shard_and_sources(tmp_path) -> None:
    manifest, launcher, artifact, evaluation, payload = _append_submission_fixture(tmp_path)
    shard_root = pathlib.Path(payload["shard"]["roots"][0])

    loaded, digest = paired_merge._validate_append_submission(  # noqa: SLF001
        manifest,
        launcher,
        shard_root=shard_root,
        repo_root=tmp_path / "repo",
        artifact_path=artifact,
        evaluation_contract=evaluation,
        start_seed=paired_merge.SHARD_STARTS[0],
    )

    assert loaded == payload
    assert len(digest) == 64


@pytest.mark.parametrize("tamper", ["launcher_sha", "shard_root", "n20_dependency"])
def test_append_submission_contract_rejects_tampering(tmp_path, tamper) -> None:
    manifest, launcher, artifact, evaluation, payload = _append_submission_fixture(tmp_path)
    shard_root = pathlib.Path(payload["shard"]["roots"][0])
    if tamper == "launcher_sha":
        payload["sources"]["launcher"]["sha256"] = "0" * 64
    elif tamper == "shard_root":
        payload["shard"]["roots"][0] = str(shard_root.with_name("wrong"))
    else:
        payload["n20_dependency"] = "afterok:1"
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="append submission"):
        paired_merge._validate_append_submission(  # noqa: SLF001
            manifest,
            launcher,
            shard_root=shard_root,
            repo_root=tmp_path / "repo",
            artifact_path=artifact,
            evaluation_contract=evaluation,
            start_seed=paired_merge.SHARD_STARTS[0],
        )


@pytest.mark.parametrize(
    ("policy", "changed_field", "changed_value"),
    [
        ("student", "policy_name", "mot_control_v1"),
        ("student", "temporal_ensemble", False),
        ("student", "arm_slice", "9:16"),
        ("student", "no_video", False),
        ("official", "policy_name", "ftp1"),
        ("official", "checkpoint_dir", "/wrong/checkpoint"),
        ("official", "action_rep", "absolute"),
        ("official", "chunk_first_n", 19),
    ],
)
def test_exact_preregistered_eval_args_reject_tampering(tmp_path, policy, changed_field, changed_value) -> None:
    root = (tmp_path / "shard").resolve()
    repo = (tmp_path / "repo").resolve()
    artifact = (tmp_path / "run/artifact.json").resolve()
    checkpoint = (tmp_path / "official/19999").resolve()
    expected = paired_merge._expected_eval_args(  # noqa: SLF001
        policy=policy,
        shard_root=root,
        repo_root=repo,
        artifact_path=artifact,
        official_checkpoint=checkpoint,
        start_seed=paired_merge.SHARD_STARTS[0],
    )
    assert paired_merge._validate_eval_args({"eval_args": expected}, expected, policy=policy)  # noqa: SLF001
    tampered = dict(expected)
    tampered[changed_field] = changed_value
    with pytest.raises(ValueError, match=f"{policy} evaluator args differ"):
        paired_merge._validate_eval_args({"eval_args": tampered}, expected, policy=policy)  # noqa: SLF001


def test_eval_arg_path_aliases_are_canonicalized_without_relaxing_nonpaths(tmp_path) -> None:
    physical = tmp_path / "physical"
    repo = physical / "repo"
    shard = physical / "paired/shard"
    run = physical / "v2"
    checkpoint = physical / "official/19999"
    for path in (repo / "UniVTAC/task_config", shard / "student", run, checkpoint):
        path.mkdir(parents=True, exist_ok=True)
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    artifact = run / "artifact.json"
    expected = paired_merge._expected_eval_args(  # noqa: SLF001
        policy="student",
        shard_root=shard,
        repo_root=repo,
        artifact_path=artifact,
        official_checkpoint=checkpoint,
        start_seed=paired_merge.SHARD_STARTS[0],
    )
    actual = dict(expected)
    for field in ("checkpoint_dir", "save_root", "task_config_path"):
        actual[field] = actual[field].replace(str(physical), str(alias), 1)

    normalized = paired_merge._validate_eval_args({"eval_args": actual}, expected, policy="student")  # noqa: SLF001

    assert normalized["checkpoint_dir"] == str(run.resolve())
    assert normalized["task_config_path"] == str((repo / "UniVTAC/task_config/demo.yml").resolve())


def test_canonical_shard_member_rejects_symlink(shard_roots, tmp_path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    member = shard_roots[0] / "evaluation_contract.json"
    member.unlink()
    member.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        paired_merge.load_shard_envelopes(shard_roots)
