from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

_ROOT = pathlib.Path(__file__).parents[1]
_LAUNCHER = _ROOT / "scripts_exp_zarr/mot_jepa/control_v2_submit_paired_n100.sh"
_SHARD = _ROOT / "scripts_exp_zarr/mot_jepa/control_v2_closedloop_paired_shard.sbatch"
_MERGE = _ROOT / "scripts_exp_zarr/mot_jepa/control_v2_merge_paired_eval.sbatch"
_STARTS = (1_000_000, 1_000_020, 1_000_040, 1_000_060, 1_000_080)


def _mock_slurm(tmp_path: pathlib.Path) -> pathlib.Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "${MOCK_SBATCH_CALLS}"\n'
        'count="$(cat "${MOCK_SBATCH_COUNTER}")"\n'
        "count=$((count + 1))\n"
        'printf "%s\\n" "${count}" > "${MOCK_SBATCH_COUNTER}"\n'
        'if [[ "${MOCK_FAIL_AT:-0}" -eq "${count}" ]]; then exit 17; fi\n'
        'printf "71%05d\\n" "${count}"\n'
    )
    sbatch.chmod(0o755)
    scancel = bin_dir / "scancel"
    scancel.write_text('#!/bin/bash\nset -euo pipefail\nprintf "%s\\n" "$*" >> "${MOCK_SCANCEL_CALLS}"\n')
    scancel.chmod(0o755)
    scontrol = bin_dir / "scontrol"
    scontrol.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "${MOCK_SCONTROL_CALLS}"\n'
        'if [[ "${MOCK_SCONTROL_FAIL:-0}" == "1" ]]; then exit 23; fi\n'
    )
    scontrol.chmod(0o755)
    return bin_dir


def _run(
    tmp_path: pathlib.Path,
    *,
    submit: bool = True,
    fail_at: int = 0,
    release_fail: bool = False,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str], list[str], pathlib.Path, pathlib.Path]:
    bin_dir = _mock_slurm(tmp_path)
    calls_path = tmp_path / "sbatch.calls"
    cancel_path = tmp_path / "scancel.calls"
    scontrol_path = tmp_path / "scontrol.calls"
    counter = tmp_path / "sbatch.counter"
    counter.write_text("0\n")
    repo = tmp_path / "repo"
    sbatch_dir = repo / "scripts_exp_zarr/mot_jepa"
    sbatch_dir.mkdir(parents=True)
    for name in ("control_v2_closedloop_paired_shard.sbatch", "control_v2_merge_paired_eval.sbatch"):
        (sbatch_dir / name).write_text(f"fixture {name}\n")
    pair_parent = tmp_path / "paired"
    manifest = repo / "logs/append.json"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "MOCK_SCANCEL_CALLS": str(cancel_path),
            "MOCK_SBATCH_COUNTER": str(counter),
            "MOCK_FAIL_AT": str(fail_at),
            "MOCK_SCONTROL_CALLS": str(scontrol_path),
            "MOCK_SCONTROL_FAIL": "1" if release_fail else "0",
            "REPO_ROOT": str(repo),
            "V2_RUN": str(tmp_path / "future_v2_run"),
            "N20_JOB": "6704613",
            "PAIR_PARENT": str(pair_parent),
            "EVAL_NODE": "gpu-h100-0044",
            "MANIFEST": str(manifest),
            "SUBMIT": "1" if submit else "0",
        }
    )
    env.update(extra_env or {})
    result = subprocess.run(["bash", str(_LAUNCHER)], env=env, text=True, capture_output=True, check=False)
    calls = calls_path.read_text().splitlines() if calls_path.exists() else []
    cancellations = cancel_path.read_text().splitlines() if cancel_path.exists() else []
    releases = scontrol_path.read_text().splitlines() if scontrol_path.exists() else []
    return result, calls, cancellations, releases, manifest, pair_parent


def test_submits_exact_five_paired_shards_and_one_cpu_merge(tmp_path: pathlib.Path) -> None:
    result, calls, cancellations, releases, manifest_path, pair_parent = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert not cancellations
    assert releases == ["release 7100001,7100002,7100003,7100004,7100005,7100006"]
    assert len(calls) == 6
    shard_jobs = [f"71{index:05d}" for index in range(1, 6)]
    for index, (call, start, _job_id) in enumerate(zip(calls[:5], _STARTS, shard_jobs, strict=True)):
        assert "control_v2_closedloop_paired_shard.sbatch" in call
        assert "--hold" in call
        assert "--dependency=afterok:6704613" in call
        assert "--partition=batch" in call
        assert "--time=04:00:00" in call
        assert "--no-requeue" in call
        assert "--nodelist=gpu-h100-0044" in call
        assert f"START_SEED={start}" in call
        assert f"PAIR_ROOT={pair_parent}/lift_bottle_s{start}_n20_%j" in call
        assert "RUN_SECONDS=5400" in call
        assert f"APPEND_MANIFEST={manifest_path}" in call
        assert f"APPEND_LAUNCHER={_LAUNCHER.resolve()}" in call
        assert index == _STARTS.index(start)

    merge = calls[5]
    assert "control_v2_merge_paired_eval.sbatch" in merge
    assert "--hold" in merge
    assert f"--dependency=afterok:{':'.join(shard_jobs)}" in merge
    assert "--partition=cpu" in merge
    assert "--time=02:00:00" in merge
    assert "--no-requeue" in merge
    assert f"PAIR_ROOT={pair_parent}/lift_bottle_s1000000_n100_%j" in merge
    for start, job_id in zip(_STARTS, shard_jobs, strict=True):
        assert f"{pair_parent}/lift_bottle_s{start}_n20_{job_id}" in merge

    payload = json.loads(manifest_path.read_text())
    assert payload["status"] == "SUBMITTED"
    assert payload["n20_job_id"] == "6704613"
    assert payload["shard"]["start_seeds"] == list(_STARTS)
    assert payload["shard"]["job_ids"] == shard_jobs
    assert payload["shard"]["time_limit"] == "04:00:00"
    assert payload["shard"]["run_seconds_per_policy"] == 5400
    assert payload["merge"]["job_id"] == "7100006"
    assert payload["merge"]["root"] == f"{pair_parent}/lift_bottle_s1000000_n100_7100006"
    assert payload["job_ids"] == [*shard_jobs, "7100006"]


@pytest.mark.parametrize(
    ("fail_at", "expected_cancel"),
    [
        (3, "7100001 7100002"),
        (6, "7100001 7100002 7100003 7100004 7100005"),
    ],
)
def test_submission_failure_cancels_every_previously_accepted_job(
    tmp_path: pathlib.Path,
    fail_at: int,
    expected_cancel: str,
) -> None:
    result, _calls, cancellations, releases, manifest, _pair_parent = _run(tmp_path, fail_at=fail_at)

    assert result.returncode == 17
    assert cancellations == [expected_cancel]
    assert not releases
    assert not manifest.exists()
    assert not pathlib.Path(f"{manifest}.lock").exists()


def test_dry_run_does_not_submit_or_write_manifest(tmp_path: pathlib.Path) -> None:
    result, calls, cancellations, releases, manifest, _pair_parent = _run(tmp_path, submit=False)

    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
    assert not calls
    assert not cancellations
    assert not releases
    assert not manifest.exists()


def test_nonqualified_node_is_rejected_before_submission(tmp_path: pathlib.Path) -> None:
    result, calls, cancellations, releases, manifest, _pair_parent = _run(
        tmp_path,
        extra_env={"EVAL_NODE": "gpu-h100-0001"},
    )

    assert result.returncode == 2
    assert "preregistered" in result.stderr
    assert not calls
    assert not cancellations
    assert not releases
    assert not manifest.exists()


def test_release_failure_cancels_all_held_jobs_and_removes_manifest(tmp_path: pathlib.Path) -> None:
    result, calls, cancellations, releases, manifest, _pair_parent = _run(tmp_path, release_fail=True)

    assert result.returncode == 23
    assert len(calls) == 6
    assert releases == ["release 7100001,7100002,7100003,7100004,7100005,7100006"]
    assert cancellations == ["7100001 7100002 7100003 7100004 7100005 7100006"]
    assert not manifest.exists()


def test_paired_scheduler_envelopes_and_post_acceptance_verification_are_fixed() -> None:
    shard = _SHARD.read_text()
    merge = _MERGE.read_text()

    assert "#SBATCH --partition=batch" in shard
    assert "#SBATCH --time=04:00:00" in shard
    assert "#SBATCH --no-requeue" in shard
    assert 'RUN_SECONDS="${RUN_SECONDS:-5400}"' in shard
    assert '"${RUN_SECONDS}" -gt 5400' in shard
    assert 'PAIR_ROOT="${PAIR_ROOT//%j/${SLURM_JOB_ID}}"' in shard
    assert "#SBATCH --partition=cpu" in merge
    assert "#SBATCH --no-requeue" in merge
    assert 'PAIR_ROOT="${PAIR_ROOT//%j/${SLURM_JOB_ID}}"' in merge
    acceptance = merge.index("mot_jepa_control_v2_acceptance.py")
    post_verify = merge.index("post-verifying combined raw records", acceptance)
    accepted_marker = merge.index('> "${PAIR_ROOT}/DONE"', post_verify)
    rejected_marker = merge.index('> "${PAIR_ROOT}/REJECTED"', post_verify)
    assert acceptance < post_verify < accepted_marker < rejected_marker
