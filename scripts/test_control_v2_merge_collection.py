from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import h5py
import pytest

from scripts import mot_jepa_control_v2_provenance as provenance
from scripts import mot_jepa_control_v2_provenance_batch as batch_provenance

_MERGE_SCRIPT = pathlib.Path(__file__).parents[1] / "scripts_exp_zarr/mot_jepa/control_v2_merge_collection_batch.sbatch"
_PROFILE_FILES = (
    "UniVTAC/task_config/mot_control_v2.yml",
    "UniVTAC/scripts/parallel_collect_data.py",
    "UniVTAC/policy/task_settings.json",
    "scripts/mot_jepa_control_v2_provenance.py",
    "scripts/mot_jepa_control_v2_provenance_batch.py",
    "scripts/mot_jepa_control_v2_qualified_runtime.py",
    "scripts_exp_zarr/mot_jepa/control_v2_collect.sbatch",
    "scripts_exp_zarr/mot_jepa/control_v2_collect_batch.sbatch",
    "scripts_exp_zarr/mot_jepa/control_v2_merge_collection_batch.sbatch",
)
_PROFILE_DIRECTORIES = ("UniVTAC/envs", "UniVTAC/assets")
_REQUIRED_HDF5 = {
    "embodiment/joint",
    "embodiment/command",
    "control/phase",
    "control/contact",
    "step",
    "observation/head/rgb",
    "tactile/left_tactile/rgb_marker",
    "tactile/right_tactile/rgb_marker",
}


def _write_profile(root: pathlib.Path, *, marker: str) -> list[pathlib.Path]:
    roots = []
    for relative in _PROFILE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{marker}:{relative}\n")
        roots.append(path)
    for relative in _PROFILE_DIRECTORIES:
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        (path / "fixture.py").write_text(f"{marker}:{relative}\n")
        roots.append(path)
    return roots


def _write_container(path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.truncate(38_711_164_928)
    return path


def _write_external_assets(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    for index in range(237):
        with (path / f"asset_{index:03d}.bin").open("wb") as stream:
            stream.truncate(429_248_620 if index == 0 else 0)
    return path


def _qualified_payload(container: pathlib.Path, external_assets: pathlib.Path, local_assets: pathlib.Path) -> dict:
    payload = {
        "schema_version": 1,
        "qualification": "univtac_common_runtime_v1",
        "common": {
            "container": {
                "path": str(container.resolve()),
                "bytes": 38_711_164_928,
                "sha256": "d8a83ddb9cf71fa37f4a39cc84a3244ec3a7f3c44419128a14eec29e39835af0",
            },
            "external_assets": {
                "path": str(external_assets.resolve()),
                "file_count": 237,
                "bytes": 429_248_620,
                "sha256": "534a909c5c09a21878d0569e52bd0fcf5137b6662511bb51ff9f38c9ae7af368",
            },
            "local_assets": {
                "path": str(local_assets.resolve()),
                "file_count": 65,
                "bytes": 58_445_582,
                "sha256": "66be4cda23ac6b11728186103f0b856273e2778f1f61a5bc65fa4a4ff57c1e67",
            },
        },
    }
    payload["qualification_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _write_hdf5(path: pathlib.Path, *, frames: int = 63) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as episode:
        for name in _REQUIRED_HDF5:
            group_name, _, dataset_name = name.rpartition("/")
            target = episode.require_group(group_name) if group_name else episode
            target.create_dataset(dataset_name, shape=(frames, 1), dtype="u1")


def _write_shard(
    root: pathlib.Path,
    *,
    rank: int,
    profile_roots: list[pathlib.Path],
    local_assets: pathlib.Path,
    container: pathlib.Path,
    external_assets: pathlib.Path,
    shard_count: int = 4,
    episode_quota: int = 1,
    collection_mode: str = "smoke",
) -> pathlib.Path:
    root.mkdir(parents=True)
    data_root = root / "raw/lift_bottle/mot_control_v2"
    parser_root = root / "parser_input"
    raw_paths = [data_root / f"{2_000_000 + rank + offset * shard_count}.hdf5" for offset in range(episode_quota)]
    _write_hdf5(raw_paths[0])
    for raw in raw_paths[1:]:
        os.link(raw_paths[0], raw)
    parser_hdf5 = parser_root / "lift_bottle/demo/hdf5"
    parser_hdf5.mkdir(parents=True)
    for raw in raw_paths:
        os.link(raw, parser_hdf5 / raw.name)

    runtime_config = root / "mot_control_v2.yml"
    runtime_config.write_text("start_seed: 2000000\n")
    qualified = root / "qualified_runtime.json"
    qualified_after = root / "qualified_runtime_after.json"
    qualified.write_text(json.dumps(_qualified_payload(container, external_assets, local_assets)))
    qualified_after.write_bytes(qualified.read_bytes())
    shard_collection = root / "SHARD_COLLECTION.json"
    shard_collection.write_text(json.dumps({"schema_version": 1, "rank": rank}))
    topology = root / "GPU_TOPOLOGY.json"
    topology.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shard_rank": rank,
                "job_id": str(100 + rank),
                "gpu_uuid": f"GPU-fixture-{rank}",
                "node": "gpu-fixture",
                "http_port": 45_600 + rank,
            }
        )
    )

    source_provenance = root / "source_provenance.json"
    source = provenance.snapshot(
        [qualified, runtime_config, *profile_roots],
        stat_only_paths=[container, external_assets],
    )
    source_provenance.write_text(json.dumps(source))
    content_sha256 = provenance.file_sha256(raw_paths[0])
    episodes = [
        {
            "path": str(raw.resolve()),
            "parser_path": str((parser_hdf5 / raw.name).resolve()),
            "logical_path": raw.name,
            "bytes": raw.stat().st_size,
            "frames": 63,
            "sha256": content_sha256,
        }
        for raw in raw_paths
    ]
    manifest = {
        "schema_version": 3,
        "status": "DONE",
        "job_id": str(100 + rank),
        "requested_episodes": episode_quota,
        "collection_mode": collection_mode,
        "global_episodes": 4 if collection_mode == "smoke" else 1000,
        "shard_rank": rank,
        "shard_count": shard_count,
        "saved_episodes": episode_quota,
        "runtime_config": str(runtime_config.resolve()),
        "runtime_config_sha256": provenance.file_sha256(runtime_config),
        "source_provenance": str(source_provenance.resolve()),
        "source_provenance_sha256": provenance.file_sha256(source_provenance),
        "source_content_sha256": source["content_sha256"],
        "qualified_runtime_sha256": provenance.file_sha256(qualified),
        "qualified_runtime_after_sha256": provenance.file_sha256(qualified_after),
        "shard_collection": str(shard_collection.resolve()),
        "shard_collection_sha256": provenance.file_sha256(shard_collection),
        "gpu_topology": str(topology.resolve()),
        "gpu_topology_sha256": provenance.file_sha256(topology),
        "data_root": str(data_root.resolve()),
        "parser_input_root": str(parser_root.resolve()),
        "episodes": episodes,
        "episode_content_sha256": provenance.collection_content_digest(episodes),
        "completed_unix": 1.0,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (root / "DONE").write_text(f"{episode_quota}\n")
    return manifest_path


def _prepare_merged_root(
    root: pathlib.Path,
    *,
    shard_manifests: list[pathlib.Path],
    profile_roots: list[pathlib.Path],
    container: pathlib.Path,
    external_assets: pathlib.Path,
) -> None:
    root.mkdir(parents=True)
    runtime_config = root / "mot_control_v2.yml"
    runtime_config.write_text("start_seed: 2000000\n")
    for name in ("qualified_runtime.json", "qualified_runtime_after.json"):
        os.link(shard_manifests[0].parent / name, root / name)
    shard_artifacts = []
    for manifest_path in shard_manifests:
        shard_root = manifest_path.parent
        shard_artifacts.extend(
            shard_root / name
            for name in (
                "DONE",
                "manifest.json",
                "source_provenance.json",
                "SHARD_COLLECTION.json",
                "GPU_TOPOLOGY.json",
            )
        )
    source = provenance.snapshot(
        [
            runtime_config,
            root / "qualified_runtime.json",
            root / "qualified_runtime_after.json",
            *profile_roots,
            *shard_artifacts,
        ],
        stat_only_paths=[container, external_assets],
    )
    (root / "source_provenance.json").write_text(json.dumps(source))


def _remove_source_profile_root(manifest_path: pathlib.Path, relative: str) -> None:
    source_path = manifest_path.parent / "source_provenance.json"
    source = json.loads(source_path.read_text())
    suffix = f"/{relative}"
    matching_roots = [record for record in source["roots"] if record["path"].endswith(suffix)]
    assert len(matching_roots) == 1
    removed_root = pathlib.Path(matching_roots[0]["path"]).resolve()
    source["roots"] = [record for record in source["roots"] if pathlib.Path(record["path"]).resolve() != removed_root]
    source["files"] = {
        path: record for path, record in source["files"].items() if pathlib.Path(path).resolve() != removed_root
    }
    source["file_count"] = len(source["files"])
    index = "".join(
        f"{path}\0{record['bytes']}\0{record['mtime_ns']}\0{record['inode']}\0{record['sha256']}\n"
        for path, record in sorted(source["files"].items())
    )
    source["content_sha256"] = hashlib.sha256(index.encode()).hexdigest()
    source_path.write_text(json.dumps(source))

    manifest = json.loads(manifest_path.read_text())
    manifest["source_provenance_sha256"] = batch_provenance.file_sha256(source_path)
    manifest["source_content_sha256"] = source["content_sha256"]
    manifest_path.write_text(json.dumps(manifest))


def _merge_python_block() -> str:
    blocks = re.findall(r"^python3 - <<'PY'\n(.*?)^PY$", _MERGE_SCRIPT.read_text(), flags=re.MULTILINE | re.DOTALL)
    assert len(blocks) == 2
    return blocks[1]


def _run_inline_merge(
    root: pathlib.Path,
    shard_manifests: list[pathlib.Path],
    *,
    collection_mode: str = "smoke",
    expected_per_shard: int = 1,
) -> subprocess.CompletedProcess[str]:
    expected_episodes = 4 if collection_mode == "smoke" else 1000
    env = os.environ.copy()
    env.update(
        {
            "COLLECTION_ROOT": str(root.resolve()),
            "DATA_ROOT": str((root / "raw/lift_bottle/mot_control_v2").resolve()),
            "PARSER_ROOT": str((root / "parser_input/lift_bottle/demo/hdf5").resolve()),
            "RUNTIME_CONFIG": str((root / "mot_control_v2.yml").resolve()),
            "SOURCE_PROVENANCE": str((root / "source_provenance.json").resolve()),
            "QUALIFIED_RUNTIME": str((root / "qualified_runtime.json").resolve()),
            "QUALIFIED_RUNTIME_AFTER": str((root / "qualified_runtime_after.json").resolve()),
            "SHARD_ROOTS": ":".join(str(path.parent.resolve()) for path in shard_manifests),
            "COLLECTION_MODE": collection_mode,
            "EXPECTED_EPISODES": str(expected_episodes),
            "EXPECTED_PER_SHARD": str(expected_per_shard),
            "EXPECTED_SHARD_COUNT": str(len(shard_manifests)),
            "SLURM_JOB_ID": "999",
        }
    )
    return subprocess.run(
        [sys.executable, "-c", _merge_python_block()],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _build_merge_fixture(
    tmp_path: pathlib.Path,
    *,
    drift: str | None = None,
    collection_mode: str = "smoke",
    shard_count: int = 4,
    episode_quota: int = 1,
):
    shared = tmp_path / "shared"
    profile = _write_profile(shared / "repo", marker="official")
    alternate_profile = _write_profile(shared / "alternate_repo", marker="changed")
    container = _write_container(shared / "univtac_full.sqsh")
    alternate_container = _write_container(shared / "alternate_univtac_full.sqsh")
    assets = _write_external_assets(shared / "tacex-assets")
    alternate_assets = _write_external_assets(shared / "alternate-tacex-assets")
    shard_manifests = [
        _write_shard(
            tmp_path / f"shard_{rank}",
            rank=rank,
            profile_roots=alternate_profile if drift == "collector_profile" and rank == 3 else profile,
            local_assets=shared / "repo/UniVTAC",
            container=alternate_container if drift == "container" and rank == 3 else container,
            external_assets=alternate_assets if drift == "external_assets" and rank == 3 else assets,
            shard_count=shard_count,
            episode_quota=episode_quota,
            collection_mode=collection_mode,
        )
        for rank in range(shard_count)
    ]
    for manifest_path in shard_manifests:
        batch_provenance.verify_collection_content(manifest_path)
    merged = tmp_path / "merged"
    _prepare_merged_root(
        merged,
        shard_manifests=shard_manifests,
        profile_roots=profile,
        container=container,
        external_assets=assets,
    )
    return merged, shard_manifests


def test_inline_merge_closes_the_same_qualified_runtime_and_source_profile(tmp_path) -> None:
    merged, shard_manifests = _build_merge_fixture(tmp_path)

    result = _run_inline_merge(merged, shard_manifests)

    assert result.returncode == 0, result.stderr
    merged_manifest = merged / "manifest.json"
    batch_provenance.verify_collection_content(merged_manifest)
    assert json.loads(merged_manifest.read_text())["shard_rank"] == -1


def test_inline_production_merge_accepts_dynamic_shard_count_and_quota(tmp_path) -> None:
    merged, shard_manifests = _build_merge_fixture(
        tmp_path,
        collection_mode="production",
        shard_count=5,
        episode_quota=200,
    )

    result = _run_inline_merge(
        merged,
        shard_manifests,
        collection_mode="production",
        expected_per_shard=200,
    )

    assert result.returncode == 0, result.stderr
    merged_manifest = merged / "manifest.json"
    batch_provenance.verify_collection_content(merged_manifest)
    manifest = json.loads(merged_manifest.read_text())
    shard_collection = json.loads((merged / "SHARD_COLLECTION.json").read_text())
    assert (manifest["shard_count"], manifest["saved_episodes"]) == (5, 1000)
    assert (shard_collection["shard_count"], shard_collection["seed_step"]) == (5, 5)
    assert [row["quota"] for row in shard_collection["shards"]] == [200] * 5
    assert len(shard_collection["saved_seeds"]) == 1000

    manifest["requested_episodes"] = 999
    merged_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not the exact requested episode set"):
        batch_provenance.verify_collection_content(merged_manifest)


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/mot_jepa_control_v2_provenance.py",
        "scripts/mot_jepa_control_v2_provenance_batch.py",
        "scripts_exp_zarr/mot_jepa/control_v2_collect.sbatch",
        "scripts_exp_zarr/mot_jepa/control_v2_collect_batch.sbatch",
    ],
)
def test_batch_verifier_rejects_omitted_required_collector_profile_root(tmp_path, relative) -> None:
    _, shard_manifests = _build_merge_fixture(tmp_path)
    manifest_path = shard_manifests[0]
    _remove_source_profile_root(manifest_path, relative)

    with pytest.raises(ValueError, match=r"lacks required profile root|one batch collector source profile"):
        batch_provenance.verify_collection_content(manifest_path)


def test_batch_verifier_rejects_merged_collection_without_batch_merge_entrypoint(tmp_path) -> None:
    merged, shard_manifests = _build_merge_fixture(tmp_path)
    result = _run_inline_merge(merged, shard_manifests)
    assert result.returncode == 0, result.stderr
    manifest_path = merged / "manifest.json"
    _remove_source_profile_root(
        manifest_path,
        "scripts_exp_zarr/mot_jepa/control_v2_merge_collection_batch.sbatch",
    )

    with pytest.raises(ValueError, match="lacks required profile root"):
        batch_provenance.verify_collection_content(manifest_path)


@pytest.mark.parametrize("drift", ["container", "external_assets", "collector_profile"])
def test_inline_merge_rejects_standalone_valid_shard_closure_drift(tmp_path, drift) -> None:
    merged, shard_manifests = _build_merge_fixture(tmp_path, drift=drift)

    result = _run_inline_merge(merged, shard_manifests)

    assert result.returncode != 0
    assert "collection shard source/container/assets" in result.stderr
