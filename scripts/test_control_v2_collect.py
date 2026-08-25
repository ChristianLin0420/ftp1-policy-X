from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess

import pytest

_COLLECTOR = pathlib.Path(__file__).parents[1] / "scripts_exp_zarr/mot_jepa/control_v2_collect_batch.sbatch"
_FROZEN_COLLECTOR = pathlib.Path(__file__).parents[1] / "scripts_exp_zarr/mot_jepa/control_v2_collect.sbatch"
_FROZEN_COLLECTOR_SHA256 = "e1035685be7cbdb67499edb555bee4d1db7790065960899b17fdca5f3f7f64d1"


def _run_contract(
    tmp_path: pathlib.Path,
    *,
    mode: str,
    episodes: int,
    global_episodes: int,
    shard_count: int,
    shard_rank: int,
) -> subprocess.CompletedProcess[str]:
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "ASSETS": str(tmp_path / "missing-assets"),
            "COLLECTION_MODE": mode,
            "COLLECTION_ROOT": str(tmp_path / "collection"),
            "EPISODES": str(episodes),
            "GLOBAL_EPISODES": str(global_episodes),
            "HTTP_PORT_BASE": "20000",
            "IMAGE": str(tmp_path / "missing-image.sqsh"),
            "REPO_ROOT": str(empty_repo),
            "SHARD_COUNT": str(shard_count),
            "SHARD_RANK": str(shard_rank),
            "SLURM_JOB_ID": "123",
            "WORKERS": "1",
        }
    )
    return subprocess.run(
        ["bash", str(_COLLECTOR)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_frozen_smoke_collector_remains_byte_exact() -> None:
    assert hashlib.sha256(_FROZEN_COLLECTOR.read_bytes()).hexdigest() == _FROZEN_COLLECTOR_SHA256


def test_batch_collector_provenance_closes_both_collector_entrypoints() -> None:
    source = _COLLECTOR.read_text()

    assert '"${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_collect.sbatch" \\' in source
    assert '"${REPO_ROOT}/scripts_exp_zarr/mot_jepa/control_v2_collect_batch.sbatch"' in source


@pytest.mark.parametrize(
    ("episodes", "shard_count", "shard_rank"),
    [(1000, 1, 0), (500, 2, 1), (250, 4, 3), (125, 8, 7), (1, 1000, 999)],
)
def test_production_accepts_any_exact_positive_factorization(
    tmp_path: pathlib.Path, episodes: int, shard_count: int, shard_rank: int
) -> None:
    result = _run_contract(
        tmp_path,
        mode="production",
        episodes=episodes,
        global_episodes=1000,
        shard_count=shard_count,
        shard_rank=shard_rank,
    )

    assert result.returncode == 2
    assert "missing committed collection config" in result.stderr


@pytest.mark.parametrize(
    ("episodes", "global_episodes", "shard_count"),
    [(250, 1000, 1), (249, 1000, 4), (1000, 999, 1), (1000, 1000, 0), (0, 1000, 1000)],
)
def test_production_rejects_nonclosing_or_nonpositive_sharding(
    tmp_path: pathlib.Path, episodes: int, global_episodes: int, shard_count: int
) -> None:
    result = _run_contract(
        tmp_path,
        mode="production",
        episodes=episodes,
        global_episodes=global_episodes,
        shard_count=shard_count,
        shard_rank=0,
    )

    assert result.returncode == 2
    assert "production V2 sharding requires" in result.stderr
    assert "missing committed collection config" not in result.stderr


def test_smoke_remains_exactly_four_one_episode_shards(tmp_path: pathlib.Path) -> None:
    valid = _run_contract(
        tmp_path,
        mode="smoke",
        episodes=1,
        global_episodes=4,
        shard_count=4,
        shard_rank=3,
    )
    invalid = _run_contract(
        tmp_path,
        mode="smoke",
        episodes=4,
        global_episodes=4,
        shard_count=1,
        shard_rank=0,
    )

    assert valid.returncode == 2
    assert "missing committed collection config" in valid.stderr
    assert invalid.returncode == 2
    assert "topology smoke sharding requires" in invalid.stderr


@pytest.mark.parametrize(("shard_count", "shard_rank"), [(1, 1), (4, -1)])
def test_collector_rejects_rank_outside_generalized_shard_range(
    tmp_path: pathlib.Path, shard_count: int, shard_rank: int
) -> None:
    result = _run_contract(
        tmp_path,
        mode="production",
        episodes=1000 // shard_count,
        global_episodes=1000,
        shard_count=shard_count,
        shard_rank=shard_rank,
    )

    assert result.returncode == 2
    assert "qualified collection requires SHARD_RANK in [0, SHARD_COUNT)" in result.stderr
