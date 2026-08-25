#!/usr/bin/env python
"""Snapshot or verify the versioned batch-sharded Control V2 source/artifact closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import stat
import tempfile
import time
from typing import Any

import h5py


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(
        path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )


def _atomic_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = pathlib.Path(raw_tmp)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def snapshot(paths: list[pathlib.Path], *, stat_only_paths: list[pathlib.Path] | None = None) -> dict:
    records = {}
    stat_only_paths = stat_only_paths or []
    roots = [(path, True) for path in paths] + [(path, False) for path in stat_only_paths]
    for root, hash_content in roots:
        for path in _files(root.resolve()):
            logical = str(path.resolve())
            if logical in records:
                continue
            stat = path.stat()
            records[logical] = {
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "inode": stat.st_ino,
                "sha256": file_sha256(path) if hash_content else None,
            }
    index = "".join(
        f"{path}\0{record['bytes']}\0{record['mtime_ns']}\0{record['inode']}\0{record['sha256']}\n"
        for path, record in sorted(records.items())
    )
    return {
        "schema_version": 2,
        "created_unix": time.time(),
        "content_sha256": hashlib.sha256(index.encode()).hexdigest(),
        "file_count": len(records),
        "roots": [{"path": str(path.resolve()), "content_hashed": hash_content} for path, hash_content in roots],
        "files": records,
    }


def verify(manifest: dict, *, hash_content: bool = True) -> None:
    errors = []
    records = manifest.get("files")
    if not isinstance(records, dict) or not records:
        raise ValueError("provenance manifest has no files")
    index = "".join(
        f"{path}\0{record['bytes']}\0{record['mtime_ns']}\0{record['inode']}\0{record['sha256']}\n"
        for path, record in sorted(records.items())
    )
    if int(manifest.get("file_count", -1)) != len(records):
        errors.append("manifest file_count differs from its records")
    if hashlib.sha256(index.encode()).hexdigest() != manifest.get("content_sha256"):
        errors.append("manifest content digest differs from its records")
    roots = manifest.get("roots")
    if roots is not None:
        if not isinstance(roots, list) or not roots:
            raise ValueError("provenance manifest roots must be a non-empty list")
        live_paths: set[str] = set()
        for root in roots:
            if not isinstance(root, dict) or "path" not in root:
                raise ValueError("invalid provenance root record")
            try:
                live_paths.update(str(path.resolve()) for path in _files(pathlib.Path(root["path"])))
            except Exception as exc:
                errors.append(f"cannot enumerate {root.get('path')}: {type(exc).__name__}: {exc}")
        expected_paths = set(records)
        if live_paths != expected_paths:
            errors.append(
                "file set changed: "
                f"missing={sorted(expected_paths - live_paths)}, added={sorted(live_paths - expected_paths)}"
            )
    for raw_path, expected in records.items():
        path = pathlib.Path(raw_path)
        try:
            stat = path.stat()
            actual_meta = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
            expected_meta = (expected["bytes"], expected["mtime_ns"], expected["inode"])
            if actual_meta != expected_meta:
                errors.append(f"metadata changed: {path}")
                continue
            if hash_content and expected["sha256"] is not None and file_sha256(path) != expected["sha256"]:
                errors.append(f"content changed: {path}")
        except Exception as exc:
            errors.append(f"cannot verify {path}: {type(exc).__name__}: {exc}")
    if errors:
        raise ValueError("provenance verification failed:\n" + "\n".join(errors))


def collection_content_digest(episodes: list[dict[str, Any]]) -> str:
    canonical = json.dumps(episodes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _canonical_collection_member(
    manifest_path: pathlib.Path,
    manifest: dict[str, Any],
    field: str,
    relative_path: str,
    *,
    directory: bool = False,
) -> pathlib.Path:
    """Resolve one schema-3 member without allowing pointer substitution or traversal."""

    declared = manifest.get(field)
    if not isinstance(declared, str) or not declared:
        raise ValueError(f"collection schema 3 lacks {field}")
    raw = pathlib.Path(declared)
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"collection {field} must be an absolute path without traversal")

    root = manifest_path.resolve().parent
    expected = root / relative_path
    actual = raw.resolve()
    # ``root`` is already physically resolved. Comparing against the un-resolved expected member
    # also rejects a symlink placed at the canonical filename which escapes the collection root.
    if actual != expected:
        raise ValueError(f"collection {field} is not the canonical in-root member: {raw}")
    if directory:
        if not expected.is_dir() or expected.is_symlink() or not stat.S_ISDIR(expected.lstat().st_mode):
            raise ValueError(f"collection declared directory is missing: {expected}")
    elif not expected.is_file() or expected.is_symlink() or not stat.S_ISREG(expected.lstat().st_mode):
        raise ValueError(f"collection declared artifact is missing: {expected}")
    return expected


_QUALIFIED_COMMON_RECORDS = {
    "container": {
        "bytes": 38_711_164_928,
        "sha256": "d8a83ddb9cf71fa37f4a39cc84a3244ec3a7f3c44419128a14eec29e39835af0",
    },
    "external_assets": {
        "file_count": 237,
        "bytes": 429_248_620,
        "sha256": "534a909c5c09a21878d0569e52bd0fcf5137b6662511bb51ff9f38c9ae7af368",
    },
    "local_assets": {
        "file_count": 65,
        "bytes": 58_445_582,
        "sha256": "66be4cda23ac6b11728186103f0b856273e2778f1f61a5bc65fa4a4ff57c1e67",
    },
}


def _qualified_runtime_content_id(path: pathlib.Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if set(payload) != {"schema_version", "qualification", "common", "qualification_sha256"}:
        raise ValueError(f"qualified runtime has an invalid field set: {path}")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError(f"qualified runtime has an invalid schema: {path}")
    if payload.get("qualification") != "univtac_common_runtime_v1":
        raise ValueError(f"collection uses the wrong runtime qualification: {path}")
    common = payload.get("common")
    if not isinstance(common, dict) or set(common) != set(_QUALIFIED_COMMON_RECORDS):
        raise ValueError(f"qualified runtime common closure is incomplete: {path}")
    for name, expected_values in _QUALIFIED_COMMON_RECORDS.items():
        record = common.get(name)
        if not isinstance(record, dict) or set(record) != {"path", *expected_values}:
            raise ValueError(f"qualified runtime {name} record is invalid: {path}")
        declared_path = pathlib.Path(record["path"]) if isinstance(record.get("path"), str) else pathlib.Path()
        if not declared_path.is_absolute() or ".." in declared_path.parts:
            raise ValueError(f"qualified runtime {name} path is invalid: {path}")
        if any(
            type(record.get(field)) is not type(value) or record.get(field) != value
            for field, value in expected_values.items()
        ):
            raise ValueError(f"qualified runtime {name} identity is not official: {path}")
    claimed = payload.pop("qualification_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(canonical).hexdigest()
    if not isinstance(claimed, str) or claimed != actual:
        raise ValueError(f"qualified runtime content id is invalid: {path}")
    payload["qualification_sha256"] = claimed
    return actual, payload


def _verify_schema3_collection_contract(
    manifest_path: pathlib.Path, manifest: dict[str, Any]
) -> dict[str, pathlib.Path]:
    """Verify the non-HDF5 half of an immutable schema-3 collection manifest."""

    expected_fields = {
        "schema_version",
        "status",
        "job_id",
        "requested_episodes",
        "collection_mode",
        "global_episodes",
        "shard_rank",
        "shard_count",
        "saved_episodes",
        "runtime_config",
        "runtime_config_sha256",
        "source_provenance",
        "source_provenance_sha256",
        "source_content_sha256",
        "qualified_runtime_sha256",
        "qualified_runtime_after_sha256",
        "shard_collection",
        "shard_collection_sha256",
        "gpu_topology",
        "gpu_topology_sha256",
        "data_root",
        "parser_input_root",
        "episodes",
        "episode_content_sha256",
        "completed_unix",
    }
    if set(manifest) != expected_fields:
        raise ValueError("collection schema 3 manifest has an invalid field set")
    if manifest.get("status") != "DONE":
        raise ValueError("collection schema 3 status is not DONE")
    for field in ("global_episodes", "shard_rank", "shard_count"):
        if type(manifest.get(field)) is not int:
            raise ValueError(f"collection schema 3 {field} must be an integer")
    mode = manifest.get("collection_mode")
    global_episodes = manifest["global_episodes"]
    shard_rank = manifest["shard_rank"]
    shard_count = manifest["shard_count"]
    requested_episodes = manifest["requested_episodes"]
    if mode not in {"smoke", "production"}:
        raise ValueError("collection schema 3 has an invalid collection_mode")
    expected_global_episodes = 4 if mode == "smoke" else 1000
    if (
        global_episodes != expected_global_episodes
        or shard_count < 1
        or (mode == "smoke" and shard_count != 4)
        or shard_rank not in {-1, *range(shard_count)}
    ):
        raise ValueError("collection schema 3 has invalid global shard dimensions")
    expected_global = requested_episodes if shard_rank == -1 else requested_episodes * shard_count
    if global_episodes != expected_global:
        raise ValueError("collection schema 3 local/global episode quotas do not close")
    job_id = manifest.get("job_id")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise ValueError("collection schema 3 job_id must be a numeric string")
    completed_unix = manifest.get("completed_unix")
    if isinstance(completed_unix, bool) or not isinstance(completed_unix, (int, float)) or completed_unix <= 0:
        raise ValueError("collection schema 3 completed_unix must be a positive number")

    runtime_config = _canonical_collection_member(manifest_path, manifest, "runtime_config", "mot_control_v2.yml")
    source_provenance = _canonical_collection_member(
        manifest_path, manifest, "source_provenance", "source_provenance.json"
    )
    shard_collection = _canonical_collection_member(
        manifest_path, manifest, "shard_collection", "SHARD_COLLECTION.json"
    )
    gpu_topology = _canonical_collection_member(manifest_path, manifest, "gpu_topology", "GPU_TOPOLOGY.json")
    data_root = _canonical_collection_member(
        manifest_path,
        manifest,
        "data_root",
        "raw/lift_bottle/mot_control_v2",
        directory=True,
    )
    parser_root = _canonical_collection_member(
        manifest_path, manifest, "parser_input_root", "parser_input", directory=True
    )
    root = manifest_path.resolve().parent
    qualified_runtime = root / "qualified_runtime.json"
    qualified_runtime_after = root / "qualified_runtime_after.json"
    for path in (qualified_runtime, qualified_runtime_after):
        if path.resolve() != path or not path.is_file() or path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"collection schema 3 lacks canonical qualified runtime artifact: {path}")

    for path, digest_field in (
        (runtime_config, "runtime_config_sha256"),
        (source_provenance, "source_provenance_sha256"),
        (shard_collection, "shard_collection_sha256"),
        (gpu_topology, "gpu_topology_sha256"),
        (qualified_runtime, "qualified_runtime_sha256"),
        (qualified_runtime_after, "qualified_runtime_after_sha256"),
    ):
        expected_digest = manifest.get(digest_field)
        if not isinstance(expected_digest, str) or file_sha256(path) != expected_digest:
            raise ValueError(f"collection declared artifact changed: {path}")
    # Separate regular files are valid, but their bytes must prove that runtime qualification did
    # not change while the collection was being generated.
    if not os.path.samefile(qualified_runtime, qualified_runtime_after) and file_sha256(
        qualified_runtime
    ) != file_sha256(qualified_runtime_after):
        raise ValueError("qualified collection runtime changed during collection")
    runtime_content_id, runtime_payload = _qualified_runtime_content_id(qualified_runtime)
    runtime_after_content_id, _ = _qualified_runtime_content_id(qualified_runtime_after)
    if runtime_content_id != runtime_after_content_id:
        raise ValueError("qualified runtime content ids changed during collection")

    source_manifest = json.loads(source_provenance.read_text())
    expected_source_fields = {"schema_version", "created_unix", "content_sha256", "file_count", "roots", "files"}
    if set(source_manifest) != expected_source_fields:
        raise ValueError("collection source provenance has an invalid schema-2 field set")
    if type(source_manifest.get("schema_version")) is not int or source_manifest["schema_version"] != 2:
        raise ValueError("collection source provenance must use integer schema_version 2")
    source_roots_payload = source_manifest.get("roots")
    if (
        not isinstance(source_roots_payload, list)
        or not source_roots_payload
        or any(
            not isinstance(record, dict)
            or set(record) != {"path", "content_hashed"}
            or not isinstance(record["path"], str)
            or type(record["content_hashed"]) is not bool
            for record in source_roots_payload
        )
    ):
        raise ValueError("collection source provenance roots are invalid")
    verify(source_manifest)
    source_content_sha256 = manifest.get("source_content_sha256")
    if not isinstance(source_content_sha256, str) or source_manifest.get("content_sha256") != source_content_sha256:
        raise ValueError("collection source-content digest differs from its provenance manifest")
    source_files = source_manifest.get("files", {})
    for path in (runtime_config, qualified_runtime):
        if str(path.resolve()) not in source_files:
            raise ValueError(f"collection source provenance does not close required artifact: {path}")
    source_roots = {
        pathlib.Path(record["path"]).resolve(): record
        for record in source_roots_payload
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    for runtime_key in ("container", "external_assets"):
        runtime_path = pathlib.Path(runtime_payload["common"][runtime_key]["path"]).resolve()
        root_record = source_roots.get(runtime_path)
        if root_record is None or root_record.get("content_hashed") is not False:
            raise ValueError(f"collection source provenance does not close qualified {runtime_key}: {runtime_path}")
    container_record = source_files.get(str(pathlib.Path(runtime_payload["common"]["container"]["path"]).resolve()))
    if (
        not isinstance(container_record, dict)
        or container_record.get("bytes") != runtime_payload["common"]["container"]["bytes"]
    ):
        raise ValueError("qualified container size differs from its source-provenance record")
    external_root = pathlib.Path(runtime_payload["common"]["external_assets"]["path"]).resolve()
    external_records = [
        record for raw_path, record in source_files.items() if pathlib.Path(raw_path).is_relative_to(external_root)
    ]
    external_identity = runtime_payload["common"]["external_assets"]
    if (
        len(external_records) != external_identity["file_count"]
        or sum(int(record.get("bytes", -1)) for record in external_records) != external_identity["bytes"]
    ):
        raise ValueError("qualified external-assets size/count differs from source provenance")

    legacy_collector_relative = pathlib.PurePosixPath("scripts_exp_zarr/mot_jepa/control_v2_collect.sbatch")
    collector_relative = pathlib.PurePosixPath("scripts_exp_zarr/mot_jepa/control_v2_collect_batch.sbatch")
    collector_roots = [path for path in source_roots if path.as_posix().endswith(f"/{collector_relative.as_posix()}")]
    if len(collector_roots) != 1:
        raise ValueError("collection source provenance does not identify one batch collector source profile")
    repo_root = collector_roots[0].parents[2]
    required_profile_roots = (
        "UniVTAC/task_config/mot_control_v2.yml",
        "UniVTAC/scripts/parallel_collect_data.py",
        "UniVTAC/envs",
        "UniVTAC/assets",
        "UniVTAC/policy/task_settings.json",
        "scripts/mot_jepa_control_v2_provenance.py",
        "scripts/mot_jepa_control_v2_provenance_batch.py",
        "scripts/mot_jepa_control_v2_qualified_runtime.py",
        legacy_collector_relative.as_posix(),
        collector_relative.as_posix(),
    )
    if shard_rank == -1:
        required_profile_roots += ("scripts_exp_zarr/mot_jepa/control_v2_merge_collection_batch.sbatch",)
    for relative in required_profile_roots:
        required_root = (repo_root / relative).resolve()
        root_record = source_roots.get(required_root)
        if root_record is None or root_record.get("content_hashed") is not True:
            raise ValueError(f"collection source provenance lacks required profile root: {required_root}")

    return {
        "runtime_config": runtime_config,
        "source_provenance": source_provenance,
        "shard_collection": shard_collection,
        "gpu_topology": gpu_topology,
        "qualified_runtime": qualified_runtime,
        "qualified_runtime_after": qualified_runtime_after,
        "data_root": data_root,
        "parser_root": parser_root,
    }


def verify_collection_content(manifest_path: pathlib.Path, *, allow_legacy: bool = False) -> dict[str, Any]:
    """Rehash the exact HDF5 file set consumed by V3 preparation."""

    manifest_path = pathlib.Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    schema_version = manifest.get("schema_version")
    schema3 = type(schema_version) is int and schema_version == 3
    if not schema3 and (not allow_legacy or schema_version is not None):
        raise ValueError("authoritative collection manifest must use integer schema_version 3")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("collection manifest has no episode records")
    requested_episodes = manifest.get("requested_episodes")
    saved_episodes = manifest.get("saved_episodes")
    if type(requested_episodes) is not int or requested_episodes < 1:
        raise ValueError("collection requested_episodes must be a positive integer")
    if type(saved_episodes) is not int or saved_episodes < 1:
        raise ValueError("collection saved_episodes must be a positive integer")
    if saved_episodes != len(episodes):
        raise ValueError("collection saved_episodes differs from its episode records")
    if requested_episodes != len(episodes):
        raise ValueError("collection is not the exact requested episode set")
    expected_fields = {"path", "parser_path", "logical_path", "bytes", "frames", "sha256"}
    if any(not isinstance(row, dict) or set(row) != expected_fields for row in episodes):
        raise ValueError("collection episode records have an invalid field set")
    for row in episodes:
        if any(not isinstance(row[field], str) or not row[field] for field in ("path", "parser_path", "logical_path")):
            raise ValueError("collection episode path fields must be non-empty strings")
        for path_field in ("path", "parser_path"):
            episode_path = pathlib.Path(row[path_field])
            if not episode_path.is_absolute() or ".." in episode_path.parts:
                raise ValueError(f"collection episode {path_field} must be absolute and traversal-free")
        if pathlib.Path(row["logical_path"]).name != row["logical_path"] or row["logical_path"] in {".", ".."}:
            raise ValueError("collection episode logical_path must be one filename")
        if type(row["bytes"]) is not int or row["bytes"] < 0 or type(row["frames"]) is not int:
            raise ValueError("collection episode bytes/frames must be integers")
        if not isinstance(row["sha256"], str) or len(row["sha256"]) != 64:
            raise ValueError("collection episode sha256 must be a 64-character string")
    for unique_field in ("path", "parser_path", "logical_path"):
        if len({row[unique_field] for row in episodes}) != len(episodes):
            raise ValueError(f"collection contains duplicate episode {unique_field} records")
    if collection_content_digest(episodes) != manifest.get("episode_content_sha256"):
        raise ValueError("collection episode-content digest differs from its records")

    contract = None
    # Legacy unit fixtures remain readable, but production schema 3 must bind every input used by
    # preparation to a canonical path beneath the collection root.
    if schema3:
        contract = _verify_schema3_collection_contract(manifest_path, manifest)

    data_root = contract["data_root"] if contract is not None else pathlib.Path(manifest["data_root"]).resolve()
    parser_root = (
        contract["parser_root"] if contract is not None else pathlib.Path(manifest["parser_input_root"]).resolve()
    )
    raw_paths = set(data_root.rglob("*.hdf5"))
    parser_paths = set(parser_root.rglob("*.hdf5"))
    expected_raw = {pathlib.Path(row["path"]) for row in episodes}
    expected_parser = {pathlib.Path(row["parser_path"]) for row in episodes}
    if raw_paths != expected_raw:
        raise ValueError("collection raw HDF5 file set changed")
    if parser_paths != expected_parser:
        raise ValueError("collection parser-input HDF5 file set changed")

    errors = []
    for row in episodes:
        raw = pathlib.Path(row["path"])
        parser = pathlib.Path(row["parser_path"])
        expected_raw_path = data_root / row["logical_path"]
        if raw != expected_raw_path:
            errors.append(f"logical/raw path mismatch: {raw}")
            continue
        if contract is not None:
            expected_parser_path = parser_root / "lift_bottle/demo/hdf5" / row["logical_path"]
            if parser != expected_parser_path:
                errors.append(f"logical/parser path mismatch: {parser}")
                continue
        try:
            if raw.is_symlink() or parser.is_symlink():
                errors.append(f"collection HDF5 member is a symlink: {raw} / {parser}")
                continue
            raw_stat = raw.lstat()
            parser_stat = parser.lstat()
            if not stat.S_ISREG(raw_stat.st_mode) or not stat.S_ISREG(parser_stat.st_mode):
                errors.append(f"collection HDF5 member is not a regular file: {raw} / {parser}")
                continue
            if (raw_stat.st_dev, raw_stat.st_ino) != (parser_stat.st_dev, parser_stat.st_ino):
                errors.append(f"parser input is not the committed HDF5 inode: {parser}")
                continue
            if raw_stat.st_size != row["bytes"]:
                errors.append(f"HDF5 byte size changed: {raw}")
                continue
            if int(row["frames"]) < 63:
                errors.append(f"HDF5 is shorter than the V3 raw minimum: {raw}")
                continue
            if contract is not None:
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
                with h5py.File(raw, "r") as episode:
                    missing = sorted(name for name in required_hdf5 if name not in episode)
                    if missing:
                        errors.append(f"HDF5 lacks V3 datasets {missing}: {raw}")
                        continue
                    frames = int(episode["embodiment/joint"].shape[0])
                    if frames != row["frames"] or any(int(episode[name].shape[0]) != frames for name in required_hdf5):
                        errors.append(f"HDF5 arrays or declared frame count are inconsistent: {raw}")
                        continue
            if file_sha256(raw) != row["sha256"]:
                errors.append(f"HDF5 content changed: {raw}")
        except Exception as exc:
            errors.append(f"cannot verify HDF5 {raw}: {type(exc).__name__}: {exc}")
    if errors:
        raise ValueError("collection-content verification failed:\n" + "\n".join(errors))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--verify", type=pathlib.Path)
    parser.add_argument(
        "--verify-stat-only",
        type=pathlib.Path,
        help="Verify the frozen file set and size/mtime/inode tree without rereading payload bytes.",
    )
    parser.add_argument(
        "--verify-collection",
        type=pathlib.Path,
        help="Rehash and verify an authoritative collection manifest and its parser hardlinks.",
    )
    parser.add_argument(
        "--stat-only",
        action="append",
        default=[],
        type=pathlib.Path,
        help="Close size/mtime/inode without hashing payload bytes (for qualified giant immutable trees).",
    )
    parser.add_argument("paths", nargs="*", type=pathlib.Path)
    args = parser.parse_args()
    modes = (args.output, args.verify, args.verify_stat_only, args.verify_collection)
    if sum(value is not None for value in modes) != 1:
        parser.error("specify exactly one snapshot or verification mode")
    if args.output is not None:
        if not args.paths:
            parser.error("snapshot mode requires at least one path")
        payload = snapshot(args.paths, stat_only_paths=args.stat_only)
        _atomic_json(args.output, payload)
        print(json.dumps({key: payload[key] for key in ("file_count", "content_sha256")}, sort_keys=True))
    elif args.verify_collection is not None:
        if args.paths or args.stat_only:
            parser.error("collection verification reads paths from the collection manifest")
        manifest = verify_collection_content(args.verify_collection)
        print(
            json.dumps(
                {
                    "verified": True,
                    "episodes": manifest["saved_episodes"],
                    "episode_content_sha256": manifest["episode_content_sha256"],
                },
                sort_keys=True,
            )
        )
    else:
        if args.paths or args.stat_only:
            parser.error("verify mode reads paths from the manifest")
        verify_path = args.verify if args.verify is not None else args.verify_stat_only
        manifest = json.loads(verify_path.read_text())
        verify(manifest, hash_content=args.verify is not None)
        print(
            json.dumps(
                {
                    "verified": True,
                    "content_sha256": manifest["content_sha256"],
                    "payload_hashes_rechecked": args.verify is not None,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
