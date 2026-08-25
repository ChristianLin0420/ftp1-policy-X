from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess

_ROOT = pathlib.Path(__file__).parents[1]
_LAUNCHER = _ROOT / "scripts_exp_zarr/mot_jepa/control_v2_submit_pipeline_batch.sh"
_ORIGINAL = _ROOT / "scripts_exp_zarr/mot_jepa/control_v2_submit_pipeline.sh"
_PREPARE = _ROOT / "scripts_exp_zarr/mot_jepa/control_v2_prepare_data.sbatch"
_ORIGINAL_SHA256 = "c2ed253b88ac8bce6b542d3e476fba9aca7daf4eff1ac95c6b28721ee8821d33"


def _mock_slurm(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "sbatch.calls"
    counter = tmp_path / "sbatch.counter"
    counter.write_text("0\n")
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "${MOCK_SBATCH_CALLS}"\n'
        'count="$(cat "${MOCK_SBATCH_COUNTER}")"\n'
        "count=$((count + 1))\n"
        'printf "%s\\n" "${count}" > "${MOCK_SBATCH_COUNTER}"\n'
        'printf "7%06d\\n" "${count}"\n'
    )
    sbatch.chmod(0o755)
    scancel = bin_dir / "scancel"
    scancel.write_text("#!/bin/bash\nexit 99\n")
    scancel.chmod(0o755)
    sacct = bin_dir / "sacct"
    sacct.write_text('#!/bin/bash\nprintf "COMPLETED\\n"\n')
    sacct.chmod(0o755)
    return bin_dir, calls


def _run_launcher(
    tmp_path: pathlib.Path,
    *,
    extra_env: dict[str, str] | None = None,
    paired_files: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[str], pathlib.Path]:
    bin_dir, calls_path = _mock_slurm(tmp_path)
    repo_root = tmp_path / "repo"
    sbatch_dir = repo_root / "scripts_exp_zarr/mot_jepa"
    sbatch_dir.mkdir(parents=True)
    for name in ("control_v2_collect_batch.sbatch", "control_v2_merge_collection_batch.sbatch"):
        (sbatch_dir / name).write_text("fixture\n")
    if paired_files:
        for name in ("control_v2_closedloop_paired_shard.sbatch", "control_v2_merge_paired_eval.sbatch"):
            (sbatch_dir / name).write_text("fixture\n")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "MOCK_SBATCH_COUNTER": str(tmp_path / "sbatch.counter"),
            "REPO_ROOT": str(repo_root),
            "RUN_ROOT": str(tmp_path / "runs"),
            "PRETRAINED_RUN": str(tmp_path / "pretrained"),
            "PIPELINE_ID": "batch_fixture",
            "SUBMIT": "1",
            "COLLECTION_NODE": "gpu-smoke",
            "COLLECTION_NODES": "gpu-a:gpu-b:gpu-c",
            "EVAL_NODE": "gpu-eval",
        }
    )
    env.update(extra_env or {})
    result = subprocess.run(["bash", str(_LAUNCHER)], env=env, text=True, capture_output=True, check=False)
    calls = calls_path.read_text().splitlines() if calls_path.exists() else []
    return result, calls, repo_root


def test_original_launcher_is_unchanged() -> None:
    assert hashlib.sha256(_ORIGINAL.read_bytes()).hexdigest() == _ORIGINAL_SHA256


def test_prepare_data_uses_explicit_collection_contract_verifier() -> None:
    source = _PREPARE.read_text()
    assert 'COLLECTION_PROVENANCE_SCRIPT="${COLLECTION_PROVENANCE_SCRIPT:-${GENERIC_PROVENANCE_SCRIPT}}"' in source
    assert source.count('"${COLLECTION_PROVENANCE_SCRIPT}" \\\n  --verify-collection') == 2
    assert (
        '  "${COLLECTION_PROVENANCE_SCRIPT}" \\\n  "${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_prepare_data.sbatch"'
        in source
    )
    assert source.index("trap _finish EXIT") < source.index('mot_jepa_control_v2_handoff.py" \\\n  --collection-root')


def test_default_production_submission_is_dynamic_batch_only_and_round_robin(tmp_path: pathlib.Path) -> None:
    result, calls, repo_root = _run_launcher(tmp_path)

    assert result.returncode == 0, result.stderr
    assert len(calls) == 33
    smoke_calls = calls[:4]
    assert all("control_v2_collect_batch.sbatch" in call for call in smoke_calls)
    assert all("--nodelist=gpu-smoke" in call and "--time=01:00:00" in call for call in smoke_calls)

    production_calls = calls[5:25]
    expected_nodes = [f"gpu-{name}" for name in ("a", "b", "c")]
    for rank, call in enumerate(production_calls):
        assert "control_v2_collect_batch.sbatch" in call
        assert "--partition=batch" in call
        assert "--time=04:00:00" in call
        assert "backfill" not in call
        assert f"--nodelist={expected_nodes[rank % len(expected_nodes)]}" in call
        assert "--dependency=afterok:7000005" in call
        assert "COLLECTION_MODE=production" in call
        assert "EPISODES=50" in call
        assert "GLOBAL_EPISODES=1000" in call
        assert f"SHARD_RANK={rank}" in call
        assert "SHARD_COUNT=20" in call

    production_merge = calls[25]
    dependency = ":".join(str(job_id) for job_id in range(7_000_006, 7_000_026))
    assert "control_v2_merge_collection_batch.sbatch" in production_merge
    assert f"--dependency=afterok:{dependency}" in production_merge
    assert production_merge.count("/shards/lift_bottle_") == 20
    assert "EXPECTED_SHARD_COUNT=20" in production_merge
    assert "EXPECTED_PER_SHARD=50" in production_merge

    prep_call = calls[26]
    assert "control_v2_prepare_data.sbatch" in prep_call
    assert f"COLLECTION_PROVENANCE_SCRIPT={repo_root}/scripts/mot_jepa_control_v2_provenance_batch.py" in prep_call

    manifest = json.loads((repo_root / "logs/control_v2_pipeline_batch_fixture.json").read_text())
    production = manifest["collection"]["production"]
    assert production["shard_count"] == 20
    assert production["episodes_per_shard"] == 50
    assert production["global_episodes"] == 1000
    assert production["partition"] == "batch"
    assert production["time_limit"] == "04:00:00"
    assert len(production["shard_job_ids"]) == 20
    assert len(production["shard_roots"]) == 20
    assert production["assigned_nodes"] == [expected_nodes[index % 3] for index in range(20)]

    train_calls = (calls[28], calls[30])
    assert all("control_v2_train.sbatch" in call for call in train_calls)
    assert all("--partition=batch" in call and "--time=04:00:00" in call for call in train_calls)
    n1_call, n20_call = calls[31:33]
    assert "control_v2_closedloop.sbatch" in n1_call
    assert "TOTAL=1" in n1_call
    assert "--partition=batch" in n1_call
    assert "--time=00:30:00" in n1_call
    assert "--no-requeue" in n1_call
    assert "control_v2_closedloop.sbatch" in n20_call
    assert "TOTAL=20" in n20_call
    assert "--partition=batch" in n20_call
    assert "--time=02:00:00" in n20_call
    assert "--no-requeue" in n20_call
    assert not any("EVAL_MODE=paired,TOTAL=100" in call for call in calls)
    assert manifest["paired_n100"]["status"] == "deferred_after_n20"
    assert manifest["paired_n100"]["shard_job_ids"] == []
    assert manifest["paired_n100"]["merge_job_id"] is None
    assert manifest["downstream_gates"] == {
        "training_partition": "batch",
        "training_time_limit": "04:00:00",
        "n1_job_id": "7000032",
        "n1_partition": "batch",
        "n1_time_limit": "00:30:00",
        "n20_job_id": "7000033",
        "n20_partition": "batch",
        "n20_time_limit": "02:00:00",
    }


def test_paired_n100_uses_five_batch_shards_and_cpu_merge(tmp_path: pathlib.Path) -> None:
    result, calls, repo_root = _run_launcher(
        tmp_path,
        paired_files=True,
        extra_env={"SUBMIT_PAIRED_N100": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert len(calls) == 39
    start_seeds = [1_000_000 + 20 * index for index in range(5)]
    shard_calls = calls[33:38]
    pair_parent = tmp_path / "runs/mot_control_v2_eval/paired100"
    expected_roots = []
    for index, (start_seed, call) in enumerate(zip(start_seeds, shard_calls, strict=True)):
        shard_job = 7_000_034 + index
        assert "control_v2_closedloop_paired_shard.sbatch" in call
        assert "--dependency=afterok:7000033" in call
        assert "--partition=batch" in call
        assert "--time=02:00:00" in call
        assert "--no-requeue" in call
        assert "--nodelist=gpu-eval" in call
        assert f"START_SEED={start_seed}" in call
        assert f"PAIR_PARENT={pair_parent}" in call
        assert "EVAL_NODE=gpu-eval" in call
        assert "TOTAL=" not in call
        expected_roots.append(str(pair_parent / f"lift_bottle_s{start_seed}_n20_{shard_job}"))

    merge_call = calls[38]
    dependency = ":".join(str(job_id) for job_id in range(7_000_034, 7_000_039))
    assert "control_v2_merge_paired_eval.sbatch" in merge_call
    assert f"--dependency=afterok:{dependency}" in merge_call
    assert f"SHARD_ROOTS={':'.join(expected_roots)}" in merge_call
    assert f"PAIR_PARENT={pair_parent}" in merge_call
    assert "--partition=" not in merge_call

    manifest = json.loads((repo_root / "logs/control_v2_pipeline_batch_fixture.json").read_text())
    paired = manifest["paired_n100"]
    assert paired["status"] == "submitted"
    assert paired["partition"] == "batch"
    assert paired["time_limit_per_shard"] == "02:00:00"
    assert paired["trials_per_shard"] == 20
    assert paired["start_seeds"] == start_seeds
    assert paired["shard_job_ids"] == [str(job_id) for job_id in range(7_000_034, 7_000_039)]
    assert paired["shard_roots"] == expected_roots
    assert paired["assigned_node"] == "gpu-eval"
    assert paired["merge_job_id"] == "7000039"
    assert paired["merge_root"] == str(pair_parent / "lift_bottle_s1000000_n100_7000039")


def test_completed_smoke_merge_is_reused_as_dependency_gate(tmp_path: pathlib.Path) -> None:
    smoke_root = tmp_path / "merged_smoke"
    smoke_root.mkdir()
    (smoke_root / "DONE").write_text("4\n")
    (smoke_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status": "DONE",
                "collection_mode": "smoke",
                "requested_episodes": 4,
                "saved_episodes": 4,
                "global_episodes": 4,
                "shard_rank": -1,
                "shard_count": 4,
                "job_id": "6699999",
            }
        )
    )

    result, calls, repo_root = _run_launcher(tmp_path, extra_env={"SMOKE_MERGED_ROOT": str(smoke_root)})

    assert result.returncode == 0, result.stderr
    assert len(calls) == 28
    production_calls = calls[:20]
    assert all("--dependency=" not in call for call in production_calls)
    assert not any("COLLECTION_MODE=smoke" in call for call in calls)
    manifest = json.loads((repo_root / "logs/control_v2_pipeline_batch_fixture.json").read_text())
    assert manifest["reused_smoke_merge_root"] == str(smoke_root)
    assert manifest["collection"]["smoke"]["merge_job_id"] == "6699999"
    assert manifest["collection"]["smoke"]["merge_root"] == str(smoke_root)


def test_legacy_smoke_shard_roots_fail_before_any_submission(tmp_path: pathlib.Path) -> None:
    repo_root = tmp_path / "repo"
    legacy_collector = repo_root / "scripts_exp_zarr/mot_jepa/control_v2_collect.sbatch"
    roots = []
    for rank in range(4):
        root = tmp_path / f"legacy_smoke_{rank}"
        root.mkdir()
        (root / "DONE").write_text("1\n")
        (root / "manifest.json").write_text("{}\n")
        (root / "source_provenance.json").write_text(
            json.dumps(
                {
                    "roots": [
                        {
                            "path": str(legacy_collector),
                            "content_hashed": True,
                        }
                    ]
                }
            )
        )
        roots.append(str(root))

    result, calls, _ = _run_launcher(tmp_path, extra_env={"SMOKE_SHARD_ROOTS": ":".join(roots)})

    assert result.returncode == 2
    assert calls == []
    assert "SMOKE_SHARD_ROOTS accepts only batch-versioned shards" in result.stderr
    assert "SMOKE_MERGED_ROOT" in result.stderr
