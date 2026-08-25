from __future__ import annotations

import hashlib
import json
import os
import pathlib

import h5py
import pytest

from scripts import mot_jepa_control_v2_provenance as provenance


def _write_schema3_collection(tmp_path):
    data_root = tmp_path / "raw/lift_bottle/mot_control_v2"
    parser_root = tmp_path / "parser_input"
    parser_hdf5 = parser_root / "lift_bottle/demo/hdf5"
    data_root.mkdir(parents=True)
    parser_hdf5.mkdir(parents=True)
    raw = data_root / "2000000.hdf5"
    required_hdf5 = {
        "embodiment/joint",
        "embodiment/command",
        "control/phase",
        "control/contact",
        "step",
        "observation/head/rgb",
        "tactile/left_tactile/rgb_marker",
        "tactile/right_tactile/rgb_marker",
    }
    with h5py.File(raw, "w") as episode:
        for name in required_hdf5:
            group_name, _, dataset_name = name.rpartition("/")
            target = episode.require_group(group_name) if group_name else episode
            target.create_dataset(dataset_name, shape=(63, 1), dtype="u1")
    parser = parser_hdf5 / raw.name
    os.link(raw, parser)

    runtime_config = tmp_path / "mot_control_v2.yml"
    runtime_config.write_text("start_seed: 2000000\n")
    container = tmp_path / "univtac_full.sqsh"
    with container.open("wb") as stream:
        stream.truncate(38_711_164_928)
    external_assets = tmp_path / "tacex-assets"
    external_assets.mkdir()
    for index in range(237):
        asset = external_assets / f"asset_{index:03d}.bin"
        with asset.open("wb") as stream:
            stream.truncate(429_248_620 if index == 0 else 0)
    qualified_runtime = tmp_path / "qualified_runtime.json"
    qualified_runtime_after = tmp_path / "qualified_runtime_after.json"
    qualified_payload = {
        "schema_version": 1,
        "qualification": "univtac_common_runtime_v1",
        "common": {
            "container": {
                "path": str(container),
                "bytes": 38_711_164_928,
                "sha256": "d8a83ddb9cf71fa37f4a39cc84a3244ec3a7f3c44419128a14eec29e39835af0",
            },
            "external_assets": {
                "path": str(external_assets),
                "file_count": 237,
                "bytes": 429_248_620,
                "sha256": "534a909c5c09a21878d0569e52bd0fcf5137b6662511bb51ff9f38c9ae7af368",
            },
            "local_assets": {
                "path": str(tmp_path),
                "file_count": 65,
                "bytes": 58_445_582,
                "sha256": "66be4cda23ac6b11728186103f0b856273e2778f1f61a5bc65fa4a4ff57c1e67",
            },
        },
    }
    qualified_payload["qualification_sha256"] = hashlib.sha256(
        json.dumps(qualified_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    qualified_runtime.write_text(json.dumps(qualified_payload))
    qualified_runtime_after.write_bytes(qualified_runtime.read_bytes())
    shard = tmp_path / "SHARD_COLLECTION.json"
    topology = tmp_path / "GPU_TOPOLOGY.json"
    shard.write_text('{"shard_rank": 0}\n')
    topology.write_text('{"gpu_uuid": "GPU-test"}\n')
    source_provenance = tmp_path / "source_provenance.json"
    repo_root = tmp_path / "repo"
    profile_files = (
        "UniVTAC/task_config/mot_control_v2.yml",
        "UniVTAC/scripts/parallel_collect_data.py",
        "UniVTAC/policy/task_settings.json",
        "scripts/mot_jepa_control_v2_provenance.py",
        "scripts/mot_jepa_control_v2_qualified_runtime.py",
        "scripts_exp_zarr/mot_jepa/control_v2_collect.sbatch",
    )
    profile_dirs = ("UniVTAC/envs", "UniVTAC/assets")
    hashed_profile_roots = []
    for relative in profile_files:
        profile_file = repo_root / relative
        profile_file.parent.mkdir(parents=True, exist_ok=True)
        profile_file.write_text(relative)
        hashed_profile_roots.append(profile_file)
    for relative in profile_dirs:
        profile_dir = repo_root / relative
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "fixture.py").write_text(relative)
        hashed_profile_roots.append(profile_dir)
    source_payload = provenance.snapshot(
        [runtime_config, qualified_runtime, *hashed_profile_roots],
        stat_only_paths=[container, external_assets],
    )
    source_provenance.write_text(json.dumps(source_payload))

    episodes = [
        {
            "path": str(raw),
            "parser_path": str(parser),
            "logical_path": raw.name,
            "bytes": raw.stat().st_size,
            "frames": 63,
            "sha256": provenance.file_sha256(raw),
        }
    ]
    manifest = {
        "schema_version": 3,
        "status": "DONE",
        "job_id": "123",
        "collection_mode": "smoke",
        "global_episodes": 4,
        "shard_rank": 0,
        "shard_count": 4,
        "requested_episodes": 1,
        "saved_episodes": 1,
        "data_root": str(data_root),
        "parser_input_root": str(parser_root),
        "episodes": episodes,
        "episode_content_sha256": provenance.collection_content_digest(episodes),
        "runtime_config": str(runtime_config),
        "runtime_config_sha256": provenance.file_sha256(runtime_config),
        "source_provenance": str(source_provenance),
        "source_provenance_sha256": provenance.file_sha256(source_provenance),
        "source_content_sha256": source_payload["content_sha256"],
        "qualified_runtime_sha256": provenance.file_sha256(qualified_runtime),
        "qualified_runtime_after_sha256": provenance.file_sha256(qualified_runtime_after),
        "shard_collection": str(shard),
        "shard_collection_sha256": provenance.file_sha256(shard),
        "gpu_topology": str(topology),
        "gpu_topology_sha256": provenance.file_sha256(topology),
        "completed_unix": 1.0,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_snapshot_supports_hashed_and_stat_only_artifacts(tmp_path) -> None:
    hashed = tmp_path / "source.py"
    large = tmp_path / "qualified.sqsh"
    hashed.write_text("source")
    large.write_bytes(b"qualified")

    manifest = provenance.snapshot([hashed], stat_only_paths=[large])

    assert manifest["files"][str(hashed.resolve())]["sha256"] == provenance.file_sha256(hashed)
    assert manifest["files"][str(large.resolve())]["sha256"] is None
    provenance.verify(manifest)


def test_stat_only_verification_still_rejects_metadata_change(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"before")
    manifest = provenance.snapshot([], stat_only_paths=[artifact])
    artifact.write_bytes(b"different-size")

    with pytest.raises(ValueError, match="metadata changed"):
        provenance.verify(manifest)


def test_metadata_only_verification_closes_tree_without_rehashing_payload(tmp_path, monkeypatch) -> None:
    store = tmp_path / "store.zarr"
    store.mkdir()
    (store / ".zgroup").write_text("metadata")
    (store / "image_chunk").write_bytes(b"pixels")
    manifest = provenance.snapshot([store])

    monkeypatch.setattr(
        provenance,
        "file_sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("payload must not be reread")),
    )
    provenance.verify(manifest, hash_content=False)


def test_verification_rejects_files_added_after_snapshot(tmp_path) -> None:
    store = tmp_path / "store.zarr"
    store.mkdir()
    (store / ".zgroup").write_text("metadata")
    manifest = provenance.snapshot([store])
    (store / "new_chunk").write_bytes(b"pixels")

    with pytest.raises(ValueError, match="file set changed"):
        provenance.verify(manifest, hash_content=False)


def test_collection_verification_rehashes_exact_parser_hardlinks(tmp_path) -> None:
    data_root = tmp_path / "raw"
    parser_root = tmp_path / "parser_input"
    data_root.mkdir()
    parser_root.mkdir()
    raw = data_root / "episode_0.hdf5"
    raw.write_bytes(b"authoritative episode")
    parser = parser_root / raw.name
    os.link(raw, parser)
    episodes = [
        {
            "path": str(raw),
            "parser_path": str(parser),
            "logical_path": raw.name,
            "bytes": raw.stat().st_size,
            "frames": 63,
            "sha256": provenance.file_sha256(raw),
        }
    ]
    manifest = {
        "requested_episodes": 1,
        "saved_episodes": 1,
        "data_root": str(data_root),
        "parser_input_root": str(parser_root),
        "episodes": episodes,
        "episode_content_sha256": provenance.collection_content_digest(episodes),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    provenance.verify_collection_content(path, allow_legacy=True)

    raw.write_bytes(b"tampered episode!!!!!")
    with pytest.raises(ValueError, match="HDF5 content changed"):
        provenance.verify_collection_content(path, allow_legacy=True)


def test_collection_verification_rejects_parser_file_set_drift(tmp_path) -> None:
    data_root = tmp_path / "raw"
    parser_root = tmp_path / "parser_input"
    data_root.mkdir()
    parser_root.mkdir()
    raw = data_root / "episode_0.hdf5"
    raw.write_bytes(b"episode")
    parser = parser_root / raw.name
    os.link(raw, parser)
    episodes = [
        {
            "path": str(raw),
            "parser_path": str(parser),
            "logical_path": raw.name,
            "bytes": raw.stat().st_size,
            "frames": 63,
            "sha256": provenance.file_sha256(raw),
        }
    ]
    manifest = {
        "requested_episodes": 1,
        "saved_episodes": 1,
        "data_root": str(data_root),
        "parser_input_root": str(parser_root),
        "episodes": episodes,
        "episode_content_sha256": provenance.collection_content_digest(episodes),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    (parser_root / "unexpected.hdf5").write_bytes(b"extra")

    with pytest.raises(ValueError, match="parser-input HDF5 file set changed"):
        provenance.verify_collection_content(path, allow_legacy=True)


def test_schema3_collection_closes_shard_and_gpu_topology_sidecars(tmp_path) -> None:
    path, _ = _write_schema3_collection(tmp_path)
    topology = tmp_path / "GPU_TOPOLOGY.json"

    provenance.verify_collection_content(path)
    topology.write_text('{"gpu_uuid": "GPU-tampered"}\n')
    with pytest.raises(ValueError, match="declared artifact changed"):
        provenance.verify_collection_content(path)


@pytest.mark.parametrize("field", ["runtime_config", "source_provenance"])
def test_schema3_collection_rejects_substituted_contract_pointer(tmp_path, field) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    canonical = tmp_path / pathlib.Path(manifest[field]).name
    substitute = tmp_path / f"substitute_{canonical.name}"
    substitute.write_bytes(canonical.read_bytes())
    manifest[field] = str(substitute)
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="not the canonical in-root member"):
        provenance.verify_collection_content(path)


def test_schema3_collection_rejects_traversing_contract_pointer(tmp_path) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    manifest["runtime_config"] = str(tmp_path / "nested" / ".." / "mot_control_v2.yml")
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="without traversal"):
        provenance.verify_collection_content(path)


@pytest.mark.parametrize("field", ["path", "parser_path"])
def test_schema3_collection_rejects_relative_episode_pointer(tmp_path, monkeypatch, field) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    manifest["episodes"][0][field] = str(pathlib.Path(manifest["episodes"][0][field]).relative_to(tmp_path))
    manifest["episode_content_sha256"] = provenance.collection_content_digest(manifest["episodes"])
    path.write_text(json.dumps(manifest))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="absolute and traversal-free"):
        provenance.verify_collection_content(path)


def test_schema3_collection_rejects_traversing_episode_pointer(tmp_path) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    raw = pathlib.Path(manifest["episodes"][0]["path"])
    manifest["episodes"][0]["path"] = str(raw.parent / "junk" / ".." / raw.name)
    manifest["episode_content_sha256"] = provenance.collection_content_digest(manifest["episodes"])
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="absolute and traversal-free"):
        provenance.verify_collection_content(path)


def test_schema3_collection_rejects_runtime_config_tamper(tmp_path) -> None:
    path, _ = _write_schema3_collection(tmp_path)
    (tmp_path / "mot_control_v2.yml").write_text("start_seed: 9\n")

    with pytest.raises(ValueError, match="declared artifact changed"):
        provenance.verify_collection_content(path)


def test_schema3_collection_rejects_source_content_id_substitution(tmp_path) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    manifest["source_content_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="source-content digest"):
        provenance.verify_collection_content(path)


def test_schema3_collection_rejects_requalified_runtime(tmp_path) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    qualified_after = tmp_path / "qualified_runtime_after.json"
    qualified_after.write_text('{"qualification":"different"}\n')
    manifest["qualified_runtime_after_sha256"] = provenance.file_sha256(qualified_after)
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="runtime changed during collection"):
        provenance.verify_collection_content(path)


def test_schema3_collection_recomputes_runtime_content_id(tmp_path) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    invalid_payload = json.loads((tmp_path / "qualified_runtime.json").read_text())
    invalid_payload["qualification_sha256"] = "0" * 64
    invalid = json.dumps(invalid_payload)
    for name, digest_field in (
        ("qualified_runtime.json", "qualified_runtime_sha256"),
        ("qualified_runtime_after.json", "qualified_runtime_after_sha256"),
    ):
        qualified = tmp_path / name
        qualified.write_text(invalid)
        manifest[digest_field] = provenance.file_sha256(qualified)
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="runtime content id is invalid"):
        provenance.verify_collection_content(path)


def test_schema3_collection_rejects_source_provenance_schema_bypass(tmp_path) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    source_path = tmp_path / "source_provenance.json"
    source = json.loads(source_path.read_text())
    source["schema_version"] = "2"
    source_path.write_text(json.dumps(source))
    manifest["source_provenance_sha256"] = provenance.file_sha256(source_path)
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="integer schema_version 2"):
        provenance.verify_collection_content(path)


def test_schema3_collection_rejects_removed_source_roots(tmp_path) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    source_path = tmp_path / "source_provenance.json"
    source = json.loads(source_path.read_text())
    source["roots"] = None
    source_path.write_text(json.dumps(source))
    manifest["source_provenance_sha256"] = provenance.file_sha256(source_path)
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="source provenance roots are invalid"):
        provenance.verify_collection_content(path)


def test_schema3_collection_rejects_external_qualified_runtime_symlink(tmp_path) -> None:
    path, _ = _write_schema3_collection(tmp_path)
    qualified_after = tmp_path / "qualified_runtime_after.json"
    external = tmp_path.parent / f"{tmp_path.name}_qualified_runtime.json"
    external.write_bytes(qualified_after.read_bytes())
    qualified_after.unlink()
    qualified_after.symlink_to(external)

    with pytest.raises(ValueError, match="canonical qualified runtime artifact"):
        provenance.verify_collection_content(path)


def test_schema3_collection_rejects_external_hdf5_symlinks(tmp_path) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    raw = pathlib.Path(manifest["episodes"][0]["path"])
    parser = pathlib.Path(manifest["episodes"][0]["parser_path"])
    external = tmp_path.parent / f"{tmp_path.name}_episode.hdf5"
    external.write_bytes(raw.read_bytes())
    raw.unlink()
    parser.unlink()
    raw.symlink_to(external)
    parser.symlink_to(external)

    with pytest.raises(ValueError, match="HDF5 member is a symlink"):
        provenance.verify_collection_content(path)


def test_schema3_collection_rejects_incomplete_source_profile(tmp_path) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    source_path = tmp_path / "source_provenance.json"
    source = provenance.snapshot(
        [tmp_path / "mot_control_v2.yml", tmp_path / "qualified_runtime.json"],
        stat_only_paths=[tmp_path / "univtac_full.sqsh", tmp_path / "tacex-assets"],
    )
    source_path.write_text(json.dumps(source))
    manifest["source_provenance_sha256"] = provenance.file_sha256(source_path)
    manifest["source_content_sha256"] = source["content_sha256"]
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="collector source profile"):
        provenance.verify_collection_content(path)


def test_schema3_collection_rejects_non_v3_hdf5(tmp_path) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    raw = pathlib.Path(manifest["episodes"][0]["path"])
    with h5py.File(raw, "w") as episode:
        episode.create_dataset("embodiment/joint", shape=(63, 1), dtype="u1")
    manifest["episodes"][0]["bytes"] = raw.stat().st_size
    manifest["episodes"][0]["sha256"] = provenance.file_sha256(raw)
    manifest["episode_content_sha256"] = provenance.collection_content_digest(manifest["episodes"])
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="HDF5 lacks V3 datasets"):
        provenance.verify_collection_content(path)


@pytest.mark.parametrize("schema_version", [None, "3", 3.0])
def test_authoritative_collection_rejects_schema_downgrade_bypass(tmp_path, schema_version) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    if schema_version is None:
        manifest.pop("schema_version")
    else:
        manifest["schema_version"] = schema_version
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="integer schema_version 3"):
        provenance.verify_collection_content(path)


def test_schema3_collection_rejects_duplicate_episode_rows(tmp_path) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    manifest["episodes"].append(dict(manifest["episodes"][0]))
    manifest["requested_episodes"] = 2
    manifest["saved_episodes"] = 2
    manifest["episode_content_sha256"] = provenance.collection_content_digest(manifest["episodes"])
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="duplicate episode"):
        provenance.verify_collection_content(path)


@pytest.mark.parametrize("field", ["bytes", "frames"])
def test_schema3_collection_rejects_string_numeric_fields(tmp_path, field) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    manifest["episodes"][0][field] = str(manifest["episodes"][0][field])
    manifest["episode_content_sha256"] = provenance.collection_content_digest(manifest["episodes"])
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="bytes/frames must be integers"):
        provenance.verify_collection_content(path)


@pytest.mark.parametrize(("field", "value"), [("requested_episodes", "1"), ("saved_episodes", True)])
def test_schema3_collection_rejects_coerced_top_level_counts(tmp_path, field, value) -> None:
    path, manifest = _write_schema3_collection(tmp_path)
    manifest[field] = value
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=f"collection {field} must be a positive integer"):
        provenance.verify_collection_content(path)
