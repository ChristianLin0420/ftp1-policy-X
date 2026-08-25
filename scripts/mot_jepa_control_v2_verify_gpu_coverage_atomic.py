#!/usr/bin/env python
# ruff: noqa: SLF001
"""Verify eight concurrent one-GPU Control V2 smokes from one atomic allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
import time
from typing import Any

from scripts import mot_jepa_control_v2_provenance_batch as provenance
from scripts import mot_jepa_control_v2_verify_gpu_coverage as base

EXPECTED_GPU_COUNT = 8
EXPECTED_CPU_COUNT = 128
EXPECTED_MEMORY_MIB = 1_024_000
_UUID_RE = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_BARRIER_FIELDS = {
    "schema_version",
    "status",
    "allocation_job_id",
    "node",
    "observed_unix",
    "numeric_step_ids",
    "slots",
}
_BARRIER_SLOT_FIELDS = {
    "slot",
    "root",
    "shard_rank",
    "http_port",
    "gpu_uuid",
    "topology_txt_sha256",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _validate_parallel_inputs(
    roots: list[pathlib.Path],
    expected_slots: list[int],
    expected_ranks: list[int],
    expected_ports: list[int],
    *,
    allocation_job_id: str,
    require_roots: bool = True,
) -> list[dict[str, Any]]:
    lengths = {len(roots), len(expected_slots), len(expected_ranks), len(expected_ports)}
    if lengths != {EXPECTED_GPU_COUNT}:
        raise ValueError("atomic verification requires exactly eight parallel root/slot/rank/port arguments")
    canonical_roots = []
    for raw_root in roots:
        root = pathlib.Path(raw_root)
        if require_roots:
            root = base._validate_root(root)
        elif not root.is_absolute() or root.resolve() != root or root.is_symlink():
            raise ValueError("--root arguments must be absolute canonical non-symlink paths")
        canonical_roots.append(root)
    if len(set(canonical_roots)) != EXPECTED_GPU_COUNT:
        raise ValueError("--root arguments must identify eight distinct canonical collection roots")
    if set(expected_slots) != set(range(EXPECTED_GPU_COUNT)):
        raise ValueError("--expected-slot arguments must be the exact set 0..7")
    if len(set(expected_ports)) != EXPECTED_GPU_COUNT:
        raise ValueError("--expected-port arguments must contain eight distinct ports")

    slots = []
    for root, slot, rank, port in zip(canonical_roots, expected_slots, expected_ranks, expected_ports, strict=True):
        if type(slot) is not int or slot not in range(EXPECTED_GPU_COUNT):
            raise ValueError("each expected slot must be an integer in [0, 8)")
        if type(rank) is not int or rank != slot % 4:
            raise ValueError(f"expected rank for atomic slot {slot} must equal {slot % 4}")
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError(f"expected port for atomic slot {slot} is invalid")
        expected_name = f"lift_bottle_{allocation_job_id}_slot{slot}"
        if root.name != expected_name:
            raise ValueError(f"atomic slot {slot} root must end in {expected_name}")
        slots.append({"slot": slot, "root": root, "shard_rank": rank, "http_port": port})
    slots.sort(key=lambda row: row["slot"])
    parents = {row["root"].parent for row in slots}
    if len(parents) != 1:
        raise ValueError("all atomic smoke roots must share one coordinator-owned parent")
    return slots


def _query_active_numeric_steps(allocation_job_id: str) -> list[str]:
    result = subprocess.run(
        ["squeue", "--steps", "-h", "-j", allocation_job_id, "-o", "%i"],
        check=True,
        capture_output=True,
        text=True,
    )
    pattern = re.compile(rf"{re.escape(allocation_job_id)}\.[0-9]+")
    steps = [line.strip() for line in result.stdout.splitlines() if pattern.fullmatch(line.strip())]
    if len(steps) != EXPECTED_GPU_COUNT or len(set(steps)) != EXPECTED_GPU_COUNT:
        raise ValueError(f"expected eight active numeric srun steps; observed {steps}")
    return sorted(steps, key=lambda value: int(value.rsplit(".", 1)[1]))


def _memory_mib(raw: str) -> int:
    match = re.fullmatch(r"([0-9]+)([KMGTP]?)", raw)
    if match is None:
        raise ValueError(f"invalid Slurm memory TRES: {raw!r}")
    value = int(match.group(1))
    unit = match.group(2)
    numerator = {"": 1, "K": 1, "M": 1024, "G": 1024**2, "T": 1024**3, "P": 1024**4}[unit]
    denominator = 1024 if unit else 1
    converted, remainder = divmod(value * numerator, denominator)
    if remainder:
        raise ValueError(f"Slurm memory TRES is not an integral MiB value: {raw!r}")
    return converted


def _validate_scheduler_record(
    scheduler: dict[str, Any], *, allocation_job_id: str, expected_node: str
) -> dict[str, Any]:
    fields = {"job_id", "state", "exit_code", "node", "alloc_tres"}
    if not isinstance(scheduler, dict) or set(scheduler) != fields:
        raise ValueError("scheduler query did not return an exact allocation record")
    expected = {
        "job_id": allocation_job_id,
        "state": "COMPLETED",
        "exit_code": "0:0",
        "node": expected_node,
    }
    if any(type(scheduler.get(key)) is not type(value) or scheduler[key] != value for key, value in expected.items()):
        raise ValueError("atomic allocation must be COMPLETED/0:0 on the exact expected node")
    raw_tres = scheduler.get("alloc_tres")
    if not isinstance(raw_tres, str):
        raise ValueError("atomic allocation AllocTRES must be a string")
    tres: dict[str, str] = {}
    for member in raw_tres.split(","):
        key, separator, value = member.partition("=")
        if not separator or key in tres:
            raise ValueError("atomic allocation AllocTRES is malformed")
        tres[key] = value
    expected_tres = {"cpu": str(EXPECTED_CPU_COUNT), "node": "1", "gres/gpu": str(EXPECTED_GPU_COUNT)}
    if any(tres.get(key) != value for key, value in expected_tres.items()):
        raise ValueError("atomic allocation does not prove exact cpu=128,node=1,gres/gpu=8")
    if "mem" not in tres or _memory_mib(tres["mem"]) != EXPECTED_MEMORY_MIB:
        raise ValueError("atomic allocation does not prove exact mem=1000G")
    return scheduler


def _query_scheduler_record(allocation_job_id: str, expected_node: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            allocation_job_id,
            "--format=JobIDRaw,State,ExitCode,NodeList,AllocTRES",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.rstrip("|").split("|") for line in result.stdout.splitlines() if line.strip()]
    matching = [row for row in rows if row and row[0] == allocation_job_id]
    if len(matching) != 1 or len(matching[0]) != 5:
        raise ValueError("sacct did not return exactly one complete atomic allocation record")
    row = matching[0]
    scheduler = {
        "job_id": allocation_job_id,
        "state": row[1].split("+", 1)[0],
        "exit_code": row[2],
        "node": row[3],
        "alloc_tres": row[4],
    }
    return _validate_scheduler_record(scheduler, allocation_job_id=allocation_job_id, expected_node=expected_node)


def _read_topology_text(path: pathlib.Path, *, expected_rank: int, expected_port: int) -> dict[str, Any]:
    raw = base._require_direct_regular(path)
    try:
        fields = raw.decode().split()
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid GPU topology text: {path}") from exc
    if len(fields) != 3:
        raise ValueError(f"GPU topology text must contain exactly three fields: {path}")
    rank_raw, gpu_uuid, port_raw = fields
    try:
        rank = int(rank_raw)
        port = int(port_raw)
    except ValueError as exc:
        raise ValueError(f"GPU topology text contains a non-integer rank or port: {path}") from exc
    if rank != expected_rank or port != expected_port:
        raise ValueError(f"GPU topology text differs from the expected slot mapping: {path}")
    if _UUID_RE.fullmatch(gpu_uuid) is None:
        raise ValueError(f"GPU topology text does not contain a physical GPU UUID: {path}")
    return {"gpu_uuid": gpu_uuid, "topology_txt_sha256": _sha256_bytes(raw)}


def _capture_concurrency_observation(
    slots: list[dict[str, Any]],
    *,
    allocation_job_id: str,
    step_query: Any = _query_active_numeric_steps,
) -> dict[str, Any]:
    parent = slots[0]["root"].parent
    parent_failures = sorted(path.name for path in parent.glob("ATOMIC_FAILED*") if os.path.lexists(path))
    if parent_failures:
        raise ValueError(f"atomic coordinator failed before the concurrency barrier: {parent_failures}")
    records = []
    for slot in slots:
        root = slot["root"]
        failures = sorted(path.name for path in root.glob("FAILED*") if os.path.lexists(path))
        if failures:
            raise ValueError(f"collector failed before the concurrency barrier: slot={slot['slot']} {failures}")
        if os.path.lexists(root / "DONE"):
            raise ValueError(f"collector committed DONE before the concurrency barrier: slot={slot['slot']}")
        topology = _read_topology_text(
            root / "GPU_TOPOLOGY.txt",
            expected_rank=slot["shard_rank"],
            expected_port=slot["http_port"],
        )
        records.append(
            {
                "slot": slot["slot"],
                "root": str(root),
                "shard_rank": slot["shard_rank"],
                "http_port": slot["http_port"],
                **topology,
            }
        )
    uuids = [record["gpu_uuid"].lower() for record in records]
    if len(set(uuids)) != EXPECTED_GPU_COUNT:
        raise ValueError("concurrency observation does not contain eight distinct physical GPU UUIDs")
    return {"numeric_step_ids": step_query(allocation_job_id), "slots": records}


def record_concurrency_barrier(
    roots: list[pathlib.Path],
    *,
    expected_node: str,
    expected_allocation_job: str,
    expected_slots: list[int],
    expected_ranks: list[int],
    expected_ports: list[int],
    output: pathlib.Path,
    timeout_seconds: int = 600,
    observation_interval_seconds: float = 0.5,
    step_query: Any = _query_active_numeric_steps,
) -> dict[str, Any]:
    """Atomically record three stable observations of all eight live numeric srun steps."""

    allocation_job_id = base._validate_job_id(expected_allocation_job, source="--expected-allocation-job")
    if not isinstance(expected_node, str) or not expected_node.strip():
        raise ValueError("--expected-node must be a non-empty node name")
    slots = _validate_parallel_inputs(
        roots,
        expected_slots,
        expected_ranks,
        expected_ports,
        allocation_job_id=allocation_job_id,
        require_roots=False,
    )
    output = pathlib.Path(output)
    expected_output = slots[0]["root"].parent / "CONCURRENCY_BARRIER.json"
    if not output.is_absolute() or output.resolve() != output or output != expected_output:
        raise ValueError("barrier output must be the canonical coordinator-parent CONCURRENCY_BARRIER.json")
    if output.exists() or output.is_symlink():
        raise ValueError("refuse existing concurrency-barrier output")
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be a positive integer")
    if not isinstance(observation_interval_seconds, (int, float)) or observation_interval_seconds < 0:
        raise ValueError("observation interval must be non-negative")

    deadline = time.monotonic() + timeout_seconds
    stable: list[dict[str, Any]] = []
    last_pending = "GPU topology sidecars are not ready"
    while time.monotonic() < deadline:
        try:
            observation = _capture_concurrency_observation(
                slots, allocation_job_id=allocation_job_id, step_query=step_query
            )
        except FileNotFoundError:
            stable.clear()
            last_pending = "GPU topology sidecars are not ready"
        except ValueError as exc:
            message = str(exc)
            if "expected eight active numeric srun steps" not in message and "missing required input" not in message:
                raise
            stable.clear()
            last_pending = message
        else:
            if stable and observation != stable[-1]:
                stable.clear()
            stable.append(observation)
            if len(stable) >= 3:
                result = {
                    "schema_version": 1,
                    "status": "VERIFIED_CONCURRENT",
                    "allocation_job_id": allocation_job_id,
                    "node": expected_node,
                    "observed_unix": time.time(),
                    **observation,
                }
                output.parent.mkdir(parents=True, exist_ok=True)
                fd, raw_tmp = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
                temporary = pathlib.Path(raw_tmp)
                try:
                    with os.fdopen(fd, "w") as stream:
                        json.dump(result, stream, indent=2, sort_keys=True)
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    precommit = _capture_concurrency_observation(
                        slots, allocation_job_id=allocation_job_id, step_query=step_query
                    )
                    if precommit != observation:
                        raise ValueError("concurrency observation changed before atomic barrier commit")
                    os.replace(temporary, output)
                finally:
                    temporary.unlink(missing_ok=True)
                return result
        time.sleep(max(float(observation_interval_seconds), 0.01))
    raise ValueError(f"concurrency barrier timed out: {last_pending}")


def _verify_concurrency_barrier(
    barrier_path: pathlib.Path,
    *,
    allocation_job_id: str,
    expected_node: str,
    slots: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    barrier_path = pathlib.Path(barrier_path)
    parent = slots[0]["root"].parent
    expected_path = parent / "CONCURRENCY_BARRIER.json"
    if not barrier_path.is_absolute() or barrier_path.resolve() != barrier_path or barrier_path != expected_path:
        raise ValueError("--concurrency-barrier must be the canonical coordinator-parent barrier path")
    barrier, raw = base._read_json(barrier_path)
    if set(barrier) != _BARRIER_FIELDS:
        raise ValueError("concurrency barrier has an invalid field set")
    expected_scalars = {
        "schema_version": 1,
        "status": "VERIFIED_CONCURRENT",
        "allocation_job_id": allocation_job_id,
        "node": expected_node,
    }
    if any(
        type(barrier.get(field)) is not type(expected) or barrier[field] != expected
        for field, expected in expected_scalars.items()
    ):
        raise ValueError("concurrency barrier identity differs from the expected atomic allocation")
    observed_unix = barrier.get("observed_unix")
    if isinstance(observed_unix, bool) or not isinstance(observed_unix, (int, float)) or observed_unix <= 0:
        raise ValueError("concurrency barrier observed_unix must be a positive number")

    step_pattern = re.compile(rf"{re.escape(allocation_job_id)}\.[0-9]+")
    step_ids = barrier.get("numeric_step_ids")
    if (
        not isinstance(step_ids, list)
        or len(step_ids) != EXPECTED_GPU_COUNT
        or len(set(step_ids)) != EXPECTED_GPU_COUNT
        or any(not isinstance(step, str) or step_pattern.fullmatch(step) is None for step in step_ids)
    ):
        raise ValueError("concurrency barrier must contain eight distinct numeric steps from the parent allocation")

    barrier_slots = barrier.get("slots")
    if not isinstance(barrier_slots, list) or len(barrier_slots) != EXPECTED_GPU_COUNT:
        raise ValueError("concurrency barrier must contain exactly eight slot observations")
    observed_uuids = []
    for expected, record in zip(slots, barrier_slots, strict=True):
        if not isinstance(record, dict) or set(record) != _BARRIER_SLOT_FIELDS:
            raise ValueError("concurrency barrier slot has an invalid field set")
        topology = _read_topology_text(
            expected["root"] / "GPU_TOPOLOGY.txt",
            expected_rank=expected["shard_rank"],
            expected_port=expected["http_port"],
        )
        expected_record = {
            "slot": expected["slot"],
            "root": str(expected["root"]),
            "shard_rank": expected["shard_rank"],
            "http_port": expected["http_port"],
            **topology,
        }
        if record != expected_record:
            raise ValueError("concurrency barrier slot mapping or topology changed after observation")
        observed_uuids.append(record["gpu_uuid"].lower())
    if len(set(observed_uuids)) != EXPECTED_GPU_COUNT:
        raise ValueError("concurrency barrier does not prove eight distinct physical GPU UUIDs")
    return barrier, _sha256_bytes(raw)


def verify_atomic_gpu_coverage(
    roots: list[pathlib.Path],
    *,
    reference_root: pathlib.Path,
    expected_node: str,
    expected_allocation_job: str,
    expected_slots: list[int],
    expected_ranks: list[int],
    expected_ports: list[int],
    concurrency_barrier: pathlib.Path,
    scheduler_query: Any = _query_scheduler_record,
) -> dict[str, Any]:
    """Verify all smoke roots and bind them to a simultaneous eight-step observation."""

    allocation_job_id = base._validate_job_id(expected_allocation_job, source="--expected-allocation-job")
    if not isinstance(expected_node, str) or not expected_node.strip():
        raise ValueError("--expected-node must be a non-empty node name")
    slots = _validate_parallel_inputs(
        roots,
        expected_slots,
        expected_ranks,
        expected_ports,
        allocation_job_id=allocation_job_id,
    )
    reference_root = base._validate_root(pathlib.Path(reference_root))
    if reference_root in {row["root"] for row in slots} or reference_root.parent == slots[0]["root"].parent:
        raise ValueError("--reference-root must be independent from the atomic smoke parent")

    parent = slots[0]["root"].parent
    failures = sorted(path.name for path in parent.glob("ATOMIC_FAILED*") if os.path.lexists(path))
    if failures:
        raise ValueError(f"atomic coordinator parent contains failure markers: {failures}")
    atomic_done_path = parent / "ATOMIC_DONE"
    atomic_done_raw = base._require_direct_regular(atomic_done_path)
    if atomic_done_raw != b"8\n":
        raise ValueError("atomic coordinator ATOMIC_DONE must contain exactly '8\\n'")
    atomic_done_sha256 = _sha256_bytes(atomic_done_raw)
    barrier, barrier_sha256 = _verify_concurrency_barrier(
        concurrency_barrier,
        allocation_job_id=allocation_job_id,
        expected_node=expected_node,
        slots=slots,
    )
    scheduler = _validate_scheduler_record(
        scheduler_query(allocation_job_id, expected_node),
        allocation_job_id=allocation_job_id,
        expected_node=expected_node,
    )
    scheduler_sha256 = _canonical_sha256(scheduler)
    reference, reference_profile = base._verify_reference(reference_root)
    if reference["job_id"] == allocation_job_id:
        raise ValueError("production reference job must be independent from the atomic allocation")

    records = []
    profiles = []
    for slot in slots:
        record, profile = base._verify_one(slot["root"], expected_node=expected_node, expected_job=allocation_job_id)
        if record["shard_rank"] != slot["shard_rank"] or record["http_port"] != slot["http_port"]:
            raise ValueError("committed smoke root differs from its expected slot/rank/port mapping")
        record = {"slot": slot["slot"], **record}
        records.append(record)
        profiles.append(profile)

    reference_profile_sha256 = _canonical_sha256(reference_profile)
    if any(_canonical_sha256(profile) != reference_profile_sha256 for profile in profiles):
        raise ValueError("source code/profile closure differs from the production reference")
    gpu_uuids = [record["gpu_uuid"].lower() for record in records]
    ports = [record["http_port"] for record in records]
    if len(set(gpu_uuids)) != EXPECTED_GPU_COUNT:
        raise ValueError("atomic smoke roots do not prove eight distinct physical GPU UUIDs")
    if len(set(ports)) != EXPECTED_GPU_COUNT:
        raise ValueError("atomic smoke roots do not use eight distinct HTTP ports")
    barrier_uuids = [row["gpu_uuid"].lower() for row in barrier["slots"]]
    if gpu_uuids != barrier_uuids:
        raise ValueError("committed GPU topology differs from the concurrent physical UUID observation")

    barrier_path = pathlib.Path(concurrency_barrier)
    if provenance.file_sha256(barrier_path) != barrier_sha256:
        raise ValueError("concurrency barrier changed during final verification")
    if provenance.file_sha256(atomic_done_path) != atomic_done_sha256:
        raise ValueError("atomic coordinator ATOMIC_DONE changed during final verification")
    for expected, barrier_record in zip(slots, barrier["slots"], strict=True):
        if provenance.file_sha256(expected["root"] / "GPU_TOPOLOGY.txt") != barrier_record["topology_txt_sha256"]:
            raise ValueError("GPU topology text changed during final verification")

    input_set = [
        reference,
        *(
            {
                "slot": row["slot"],
                "root": row["root"],
                "job_id": row["job_id"],
                "input_sha256": row["input_sha256"],
            }
            for row in records
        ),
        {"concurrency_barrier": str(barrier_path), "sha256": barrier_sha256},
        {"atomic_done": str(atomic_done_path), "sha256": atomic_done_sha256},
        {"scheduler": scheduler, "sha256": scheduler_sha256},
    ]
    return {
        "schema_version": 1,
        "status": "VERIFIED",
        "mode": "atomic_8gpu",
        "allocation_job_id": allocation_job_id,
        "node": expected_node,
        "expected_count": EXPECTED_GPU_COUNT,
        "gpu_uuids": [row["gpu_uuid"] for row in records],
        "http_ports": ports,
        "source_profile_sha256": reference_profile_sha256,
        "barrier_sha256": barrier_sha256,
        "atomic_done_sha256": atomic_done_sha256,
        "scheduler_sha256": scheduler_sha256,
        "input_set_sha256": _canonical_sha256(input_set),
        "scheduler": scheduler,
        "reference": reference,
        "slots": records,
    }


def _add_common_slot_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--expected-allocation-job", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--root", action="append", required=True, type=pathlib.Path)
    parser.add_argument("--expected-slot", action="append", required=True, type=int)
    parser.add_argument("--expected-rank", action="append", required=True, type=int)
    parser.add_argument("--expected-port", action="append", required=True, type=int)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recorder = commands.add_parser("record-barrier")
    _add_common_slot_arguments(recorder)
    recorder.add_argument("--timeout-seconds", default=600, type=int)
    verifier = commands.add_parser("verify")
    _add_common_slot_arguments(verifier)
    verifier.add_argument("--reference-root", required=True, type=pathlib.Path)
    verifier.add_argument("--concurrency-barrier", required=True, type=pathlib.Path)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "record-barrier":
            payload = record_concurrency_barrier(
                args.root,
                expected_node=args.expected_node,
                expected_allocation_job=args.expected_allocation_job,
                expected_slots=args.expected_slot,
                expected_ranks=args.expected_rank,
                expected_ports=args.expected_port,
                output=args.output,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("refuse existing atomic GPU-coverage output")
            payload = verify_atomic_gpu_coverage(
                args.root,
                reference_root=args.reference_root,
                expected_node=args.expected_node,
                expected_allocation_job=args.expected_allocation_job,
                expected_slots=args.expected_slot,
                expected_ranks=args.expected_rank,
                expected_ports=args.expected_port,
                concurrency_barrier=args.concurrency_barrier,
            )
            base._atomic_json(args.output, payload)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"atomic GPU coverage verification failed: {exc}\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
