#!/usr/bin/env python
"""Fail-closed verification of one node's eight physical Control V2 GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import tempfile
from typing import Any

from scripts import mot_jepa_control_v2_provenance_batch as provenance

EXPECTED_GPU_COUNT = 8
_JOB_ID_RE = re.compile(r"[1-9][0-9]*")
_GPU_UUID_RE = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_TOPOLOGY_FIELDS = {"schema_version", "job_id", "node", "shard_rank", "gpu_uuid", "http_port"}
_SHARD_FIELDS = {
    "schema_version",
    "workers",
    "shard_rank",
    "shard_count",
    "global_episodes",
    "gpu_topology",
    "http_port_base",
    "episodes",
}
_MANIFEST_FIELDS = {
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
_SOURCE_FIELDS = {"schema_version", "created_unix", "content_sha256", "file_count", "roots", "files"}
HASHED_INPUTS = (
    "DONE",
    "manifest.json",
    "GPU_TOPOLOGY.json",
    "SHARD_COLLECTION.json",
    "mot_control_v2.yml",
    "source_provenance.json",
    "qualified_runtime.json",
    "qualified_runtime_after.json",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(raw_tmp)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_direct_regular(path: pathlib.Path) -> bytes:
    """Read a direct regular file and reject replacement while it is read."""

    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"missing required input: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"required input is not a direct regular file: {path}")
    payload = path.read_bytes()
    after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"required input changed while being read: {path}")
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: pathlib.Path) -> tuple[dict[str, Any], bytes]:
    raw = _require_direct_regular(path)
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON input: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain one object: {path}")
    return payload, raw


def _validate_root(raw_root: pathlib.Path) -> pathlib.Path:
    if not raw_root.is_absolute():
        raise ValueError(f"collection root must be absolute: {raw_root}")
    if raw_root.resolve() != raw_root or raw_root.is_symlink():
        raise ValueError(f"collection root must be canonical and non-symlinked: {raw_root}")
    try:
        mode = raw_root.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"collection root does not exist: {raw_root}") from exc
    if not stat.S_ISDIR(mode):
        raise ValueError(f"collection root is not a directory: {raw_root}")
    return raw_root


def _infer_job_id(root: pathlib.Path) -> str:
    match = re.search(r"(?:^|_)([1-9][0-9]*)$", root.name)
    if match is None:
        raise ValueError(f"cannot infer job ID from collection root basename: {root}")
    return match.group(1)


def _validate_job_id(job_id: str, *, source: str) -> str:
    if not isinstance(job_id, str) or _JOB_ID_RE.fullmatch(job_id) is None:
        raise ValueError(f"{source} must be a canonical positive numeric job ID")
    return job_id


def _markers_absent(root: pathlib.Path) -> None:
    stale = sorted(
        path.name for prefix in ("FAILED", "HEARTBEAT") for path in root.glob(f"{prefix}*") if os.path.lexists(path)
    )
    if stale:
        raise ValueError(f"collection root contains failure/running markers: {root}: {stale}")


def _validate_topology(
    topology: dict[str, Any],
    *,
    expected_job: str,
    expected_node: str,
    shard_rank: int,
) -> tuple[str, int]:
    if set(topology) != _TOPOLOGY_FIELDS:
        raise ValueError("GPU_TOPOLOGY.json has an invalid field set")
    if type(topology.get("schema_version")) is not int or topology["schema_version"] != 1:
        raise ValueError("GPU_TOPOLOGY.json must use integer schema_version 1")
    if topology.get("job_id") != expected_job:
        raise ValueError("GPU_TOPOLOGY.json job_id differs from the expected job")
    if topology.get("node") != expected_node:
        raise ValueError("GPU_TOPOLOGY.json node differs from --expected-node")
    if type(topology.get("shard_rank")) is not int or topology["shard_rank"] != shard_rank:
        raise ValueError("GPU_TOPOLOGY.json shard_rank differs from manifest.json")
    gpu_uuid = topology.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or _GPU_UUID_RE.fullmatch(gpu_uuid) is None:
        raise ValueError("GPU_TOPOLOGY.json gpu_uuid is not a physical GPU- UUID")
    http_port = topology.get("http_port")
    if type(http_port) is not int or not 1024 <= http_port <= 65535:
        raise ValueError("GPU_TOPOLOGY.json http_port is invalid")
    return gpu_uuid, http_port


def _validate_shard_sidecar(
    shard: dict[str, Any],
    *,
    topology: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if set(shard) != _SHARD_FIELDS:
        raise ValueError("SHARD_COLLECTION.json has an invalid field set")
    expected_scalars = {
        "schema_version": 1,
        "workers": 1,
        "shard_rank": manifest["shard_rank"],
        "shard_count": manifest["shard_count"],
        "global_episodes": manifest["global_episodes"],
        "http_port_base": topology["http_port"],
    }
    if any(type(shard.get(key)) is not int or shard[key] != value for key, value in expected_scalars.items()):
        raise ValueError("SHARD_COLLECTION.json does not describe the exact one-worker smoke shard")
    if shard.get("gpu_topology") != topology:
        raise ValueError("SHARD_COLLECTION.json embeds a different GPU topology")
    episodes = shard.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 1 or not isinstance(episodes[0], dict):
        raise ValueError("SHARD_COLLECTION.json must contain exactly one episode")


def _is_permitted_job_owned(path: pathlib.Path, root: pathlib.Path) -> bool:
    permitted_files = {root / "qualified_runtime.json", root / "mot_control_v2.yml"}
    shard_configs = root / "shard_configs"
    return path in permitted_files or path == shard_configs or path.is_relative_to(shard_configs)


def _verify_source_provenance(source: dict[str, Any]) -> None:
    if set(source) != _SOURCE_FIELDS:
        raise ValueError("source provenance has an invalid schema-2 field set")
    if type(source.get("schema_version")) is not int or source["schema_version"] != 2:
        raise ValueError("source provenance must use integer schema_version 2")
    if type(source.get("file_count")) is not int or source["file_count"] < 1:
        raise ValueError("source provenance file_count must be a positive integer")
    if not isinstance(source.get("content_sha256"), str) or len(source["content_sha256"]) != 64:
        raise ValueError("source provenance content_sha256 must be a 64-character string")
    roots = source.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("source provenance roots must be a non-empty list")
    if any(
        not isinstance(record, dict)
        or set(record) != {"path", "content_hashed"}
        or not isinstance(record["path"], str)
        or type(record["content_hashed"]) is not bool
        for record in roots
    ):
        raise ValueError("source provenance roots are invalid")
    if not isinstance(source.get("files"), dict) or not source["files"]:
        raise ValueError("source provenance files must be a non-empty object")
    provenance.verify(source)


def _shared_source_profile(source: dict[str, Any], *, root: pathlib.Path) -> dict[str, Any]:
    """Remove only job-owned runtime/config records before cross-root comparison."""

    shared_roots = []
    job_owned_roots = []
    for record in source["roots"]:
        path = pathlib.Path(record["path"]).resolve()
        if path.is_relative_to(root):
            if not _is_permitted_job_owned(path, root):
                raise ValueError(f"source provenance contains an unexpected job-owned root: {path}")
            job_owned_roots.append(
                {"relative_path": path.relative_to(root).as_posix(), "content_hashed": record["content_hashed"]}
            )
            continue
        shared_roots.append(record)

    shared_files: dict[str, Any] = {}
    job_owned_files = []
    for raw_path, record in source["files"].items():
        path = pathlib.Path(raw_path).resolve()
        if path.is_relative_to(root):
            if not _is_permitted_job_owned(path, root):
                raise ValueError(f"source provenance contains an unexpected job-owned file: {path}")
            job_owned_files.append(path.relative_to(root).as_posix())
            continue
        shared_files[raw_path] = record
    if not shared_roots or not shared_files:
        raise ValueError("source provenance has no shared source/profile closure")
    return {
        "schema_version": source["schema_version"],
        "shared_root_count": len(shared_roots),
        "shared_file_count": len(shared_files),
        "shared_roots": sorted(shared_roots, key=lambda row: row["path"]),
        "shared_files": shared_files,
        "job_owned_roots": sorted(job_owned_roots, key=lambda row: row["relative_path"]),
        "job_owned_files": sorted(job_owned_files),
    }


def _canonical_manifest_file(root: pathlib.Path, manifest: dict[str, Any], field: str, name: str) -> pathlib.Path:
    expected = root / name
    declared = manifest.get(field)
    if not isinstance(declared, str):
        raise ValueError(f"reference manifest {field} is not the canonical in-root member")
    declared_path = pathlib.Path(declared)
    if not declared_path.is_absolute() or ".." in declared_path.parts:
        raise ValueError(f"reference manifest {field} is not the canonical in-root member")
    try:
        resolved = declared_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"reference manifest {field} is not the canonical in-root member") from exc
    if resolved != expected:
        raise ValueError(f"reference manifest {field} is not the canonical in-root member")
    _require_direct_regular(expected)
    return expected


def _verify_reference(root: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a previously accepted 50-episode reference without rereading its HDF5 payloads."""

    _markers_absent(root)
    if _require_direct_regular(root / "DONE") != b"50\n":
        raise ValueError(f"reference DONE must contain exactly '50\\n': {root / 'DONE'}")
    manifest, _ = _read_json(root / "manifest.json")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("reference manifest.json has an invalid schema-3 field set")
    expected_manifest = {
        "schema_version": 3,
        "status": "DONE",
        "collection_mode": "production",
        "requested_episodes": 50,
        "saved_episodes": 50,
        "global_episodes": 1000,
        "shard_count": 20,
    }
    for field, expected in expected_manifest.items():
        if type(manifest.get(field)) is not type(expected) or manifest[field] != expected:
            raise ValueError(f"reference manifest.json {field} must equal {expected!r}")
    completed_unix = manifest.get("completed_unix")
    if isinstance(completed_unix, bool) or not isinstance(completed_unix, (int, float)) or completed_unix <= 0:
        raise ValueError("reference manifest.json completed_unix must be a positive number")
    if type(manifest.get("shard_rank")) is not int or manifest["shard_rank"] not in range(20):
        raise ValueError("reference manifest.json shard_rank must be in [0, 20)")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 50 or any(not isinstance(row, dict) for row in episodes):
        raise ValueError("reference manifest.json must contain exactly 50 episode records")
    if provenance.collection_content_digest(episodes) != manifest.get("episode_content_sha256"):
        raise ValueError("reference manifest episode-content digest differs from its records")
    job_id = _infer_job_id(root)
    if manifest.get("job_id") != job_id:
        raise ValueError("reference manifest.json job_id differs from its root-inferred job")

    artifacts = {
        "runtime_config": _canonical_manifest_file(root, manifest, "runtime_config", "mot_control_v2.yml"),
        "source_provenance": _canonical_manifest_file(root, manifest, "source_provenance", "source_provenance.json"),
        "shard_collection": _canonical_manifest_file(root, manifest, "shard_collection", "SHARD_COLLECTION.json"),
        "gpu_topology": _canonical_manifest_file(root, manifest, "gpu_topology", "GPU_TOPOLOGY.json"),
    }
    digest_fields = {
        "runtime_config": "runtime_config_sha256",
        "source_provenance": "source_provenance_sha256",
        "shard_collection": "shard_collection_sha256",
        "gpu_topology": "gpu_topology_sha256",
    }
    for name, digest_field in digest_fields.items():
        if manifest.get(digest_field) != provenance.file_sha256(artifacts[name]):
            raise ValueError(f"reference manifest digest changed for {artifacts[name]}")

    qualified_before = _require_direct_regular(root / "qualified_runtime.json")
    qualified_after = _require_direct_regular(root / "qualified_runtime_after.json")
    if qualified_before != qualified_after:
        raise ValueError("reference qualified runtime before/after artifacts are not byte-identical")
    if manifest.get("qualified_runtime_sha256") != _sha256_bytes(qualified_before):
        raise ValueError("reference qualified_runtime.json digest differs from manifest")
    if manifest.get("qualified_runtime_after_sha256") != _sha256_bytes(qualified_after):
        raise ValueError("reference qualified_runtime_after.json digest differs from manifest")

    source, _ = _read_json(artifacts["source_provenance"])
    _verify_source_provenance(source)
    if manifest.get("source_content_sha256") != source.get("content_sha256"):
        raise ValueError("reference source-content digest differs from source provenance")
    _markers_absent(root)
    if _require_direct_regular(root / "DONE") != b"50\n":
        raise ValueError("reference DONE changed during verification")
    record = {
        "root": str(root),
        "job_id": job_id,
        "input_sha256": {name: provenance.file_sha256(root / name) for name in HASHED_INPUTS},
    }
    return record, _shared_source_profile(source, root=root)


def _verify_one(root: pathlib.Path, *, expected_node: str, expected_job: str) -> tuple[dict[str, Any], dict]:
    _markers_absent(root)
    done = _require_direct_regular(root / "DONE")
    if done != b"1\n":
        raise ValueError(f"DONE must contain exactly '1\\n': {root / 'DONE'}")

    manifest, _ = _read_json(root / "manifest.json")
    expected_manifest = {
        "schema_version": 3,
        "status": "DONE",
        "collection_mode": "smoke",
        "requested_episodes": 1,
        "saved_episodes": 1,
    }
    for field, expected in expected_manifest.items():
        if type(manifest.get(field)) is not type(expected) or manifest[field] != expected:
            raise ValueError(f"manifest.json {field} must equal {expected!r}")
    if manifest.get("job_id") != expected_job:
        raise ValueError("manifest.json job_id differs from the expected job")
    for field in ("global_episodes", "shard_rank", "shard_count"):
        if type(manifest.get(field)) is not int:
            raise ValueError(f"manifest.json {field} must be an integer")
    if manifest["global_episodes"] != 4 or manifest["shard_count"] != 4 or manifest["shard_rank"] not in range(4):
        raise ValueError("manifest.json does not describe an exact Control V2 smoke shard")

    topology, _ = _read_json(root / "GPU_TOPOLOGY.json")
    gpu_uuid, http_port = _validate_topology(
        topology,
        expected_job=expected_job,
        expected_node=expected_node,
        shard_rank=manifest.get("shard_rank"),
    )
    shard, _ = _read_json(root / "SHARD_COLLECTION.json")
    _validate_shard_sidecar(shard, topology=topology, manifest=manifest)

    qualified_before = _require_direct_regular(root / "qualified_runtime.json")
    qualified_after = _require_direct_regular(root / "qualified_runtime_after.json")
    if qualified_before != qualified_after:
        raise ValueError("qualified runtime before/after artifacts are not byte-identical")

    source, _ = _read_json(root / "source_provenance.json")
    _verify_source_provenance(source)
    verified_manifest = provenance.verify_collection_content(root / "manifest.json")
    if verified_manifest != manifest:
        raise ValueError("manifest.json changed during full collection verification")

    _markers_absent(root)
    if _require_direct_regular(root / "DONE") != done:
        raise ValueError("DONE changed during verification")
    input_sha256 = {name: provenance.file_sha256(root / name) for name in HASHED_INPUTS}
    episode = manifest["episodes"][0]
    record = {
        "root": str(root),
        "job_id": expected_job,
        "node": expected_node,
        "shard_rank": manifest["shard_rank"],
        "gpu_uuid": gpu_uuid,
        "http_port": http_port,
        "episode": {"logical_path": episode["logical_path"], "sha256": episode["sha256"]},
        "input_sha256": input_sha256,
    }
    return record, _shared_source_profile(source, root=root)


def verify_gpu_coverage(
    roots: list[pathlib.Path],
    *,
    reference_root: pathlib.Path,
    expected_node: str,
    expected_jobs: list[str] | None = None,
    expected_count: int = EXPECTED_GPU_COUNT,
) -> dict[str, Any]:
    """Verify eight independent smoke roots and return their signed coverage envelope."""

    if type(expected_count) is not int or expected_count != EXPECTED_GPU_COUNT:
        raise ValueError(f"expected_count must be exactly {EXPECTED_GPU_COUNT}")
    if len(roots) != expected_count:
        raise ValueError(f"expected exactly {expected_count} --root arguments; got {len(roots)}")
    if not isinstance(expected_node, str) or not expected_node.strip():
        raise ValueError("--expected-node must be a non-empty node name")

    canonical_reference = _validate_root(pathlib.Path(reference_root))
    canonical_roots = [_validate_root(pathlib.Path(root)) for root in roots]
    if len(set(canonical_roots)) != expected_count:
        raise ValueError("--root arguments must identify eight distinct collection roots")
    if canonical_reference in canonical_roots:
        raise ValueError("--reference-root must be independent from the eight smoke roots")
    if expected_jobs is None:
        jobs = [_infer_job_id(root) for root in canonical_roots]
    else:
        if len(expected_jobs) != expected_count:
            raise ValueError(f"expected exactly {expected_count} --expected-job arguments")
        jobs = [_validate_job_id(job, source="--expected-job") for job in expected_jobs]
    if len(set(jobs)) != expected_count:
        raise ValueError("expected job IDs must be distinct")

    reference, reference_profile = _verify_reference(canonical_reference)
    if reference["job_id"] in jobs:
        raise ValueError("--reference-root job must be independent from the eight smoke jobs")

    records = []
    profiles = []
    for root, job in zip(canonical_roots, jobs, strict=True):
        record, profile = _verify_one(root, expected_node=expected_node, expected_job=job)
        records.append(record)
        profiles.append(profile)

    reference_profile_digest = _canonical_sha256(reference_profile)
    profile_digests = [_canonical_sha256(profile) for profile in profiles]
    if any(digest != reference_profile_digest for digest in profile_digests):
        raise ValueError("source code/profile closure differs from the production reference")
    gpu_ids = [record["gpu_uuid"].lower() for record in records]
    ports = [record["http_port"] for record in records]
    if len(set(gpu_ids)) != expected_count:
        raise ValueError("smoke roots do not prove eight distinct physical GPU UUIDs")
    if len(set(ports)) != expected_count:
        raise ValueError("smoke roots do not use eight distinct HTTP ports")

    records.sort(key=lambda row: int(row["job_id"]))
    input_set = [
        reference,
        *({"root": row["root"], "job_id": row["job_id"], "input_sha256": row["input_sha256"]} for row in records),
    ]
    return {
        "schema_version": 1,
        "status": "VERIFIED",
        "expected_count": expected_count,
        "expected_node": expected_node,
        "job_ids": [row["job_id"] for row in records],
        "gpu_uuids": [row["gpu_uuid"] for row in records],
        "http_ports": [row["http_port"] for row in records],
        "source_profile_sha256": reference_profile_digest,
        "input_set_sha256": _canonical_sha256(input_set),
        "reference": reference,
        "roots": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=pathlib.Path)
    parser.add_argument("--reference-root", required=True, type=pathlib.Path)
    parser.add_argument("--expected-count", default=EXPECTED_GPU_COUNT, type=int)
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--expected-job", action="append")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    try:
        payload = verify_gpu_coverage(
            args.root,
            reference_root=args.reference_root,
            expected_node=args.expected_node,
            expected_jobs=args.expected_job,
            expected_count=args.expected_count,
        )
        _atomic_json(args.output, payload)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"GPU coverage verification failed: {exc}\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
