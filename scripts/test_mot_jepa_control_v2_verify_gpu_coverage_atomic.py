from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from scripts import mot_jepa_control_v2_provenance_batch as provenance
from scripts import mot_jepa_control_v2_verify_gpu_coverage_atomic as atomic
from scripts.test_control_v2_merge_collection import _write_container
from scripts.test_control_v2_merge_collection import _write_external_assets
from scripts.test_control_v2_merge_collection import _write_profile
from scripts.test_control_v2_merge_collection import _write_shard

_NODE = "gpu-h100-0284"
_JOB_ID = "91111"


def _gpu_uuid(index: int) -> str:
    return f"GPU-{index + 1:08x}-0000-0000-0000-{index + 1:012x}"


def _rewrite_topology(
    root: pathlib.Path,
    *,
    job_id: str,
    index: int,
    port: int,
    gpu_uuid: str | None = None,
) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    uuid = gpu_uuid or _gpu_uuid(index)
    topology = {
        "schema_version": 1,
        "job_id": job_id,
        "node": _NODE,
        "shard_rank": manifest["shard_rank"],
        "gpu_uuid": uuid,
        "http_port": port,
    }
    topology_path = root / "GPU_TOPOLOGY.json"
    topology_path.write_text(json.dumps(topology))
    (root / "GPU_TOPOLOGY.txt").write_text(f"{manifest['shard_rank']} {uuid} {port}\n")
    shard = {
        "schema_version": 1,
        "workers": 1,
        "shard_rank": manifest["shard_rank"],
        "shard_count": manifest["shard_count"],
        "global_episodes": manifest["global_episodes"],
        "gpu_topology": topology,
        "http_port_base": port,
        "episodes": [{"fixture": index}],
    }
    shard_path = root / "SHARD_COLLECTION.json"
    shard_path.write_text(json.dumps(shard))
    manifest["job_id"] = job_id
    manifest["gpu_topology_sha256"] = provenance.file_sha256(topology_path)
    manifest["shard_collection_sha256"] = provenance.file_sha256(shard_path)
    manifest_path.write_text(json.dumps(manifest))


def _write_fixture(
    tmp_path: pathlib.Path,
    *,
    duplicate_uuid_at: int | None = None,
    split_profile_at: int | None = None,
) -> tuple[list[pathlib.Path], pathlib.Path, pathlib.Path, list[int]]:
    shared = tmp_path / "shared"
    profile = _write_profile(shared / "repo", marker="common")
    alternate_profile = _write_profile(shared / "alternate_repo", marker="different")
    container = _write_container(shared / "univtac_full.sqsh")
    external_assets = _write_external_assets(shared / "tacex-assets")
    local_assets = shared / "local-assets"
    parent = (tmp_path / "atomic-parent").resolve()
    ports = [32_000 + slot for slot in range(atomic.EXPECTED_GPU_COUNT)]
    roots = []
    for slot in range(atomic.EXPECTED_GPU_COUNT):
        root = parent / f"lift_bottle_{_JOB_ID}_slot{slot}"
        _write_shard(
            root,
            rank=slot % 4,
            profile_roots=alternate_profile if split_profile_at == slot else profile,
            local_assets=local_assets,
            container=container,
            external_assets=external_assets,
        )
        uuid = _gpu_uuid(0) if duplicate_uuid_at == slot else _gpu_uuid(slot)
        _rewrite_topology(root, job_id=_JOB_ID, index=slot, port=ports[slot], gpu_uuid=uuid)
        roots.append(root)

    reference = (tmp_path / "reference/lift_bottle_92999").resolve()
    _write_shard(
        reference,
        rank=0,
        profile_roots=profile,
        local_assets=local_assets,
        container=container,
        external_assets=external_assets,
        shard_count=20,
        episode_quota=50,
        collection_mode="production",
    )
    _rewrite_topology(reference, job_id="92999", index=99, port=33_999)
    barrier = parent / "CONCURRENCY_BARRIER.json"
    barrier.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "VERIFIED_CONCURRENT",
                "allocation_job_id": _JOB_ID,
                "node": _NODE,
                "observed_unix": 1.0,
                "numeric_step_ids": [f"{_JOB_ID}.{slot}" for slot in range(atomic.EXPECTED_GPU_COUNT)],
                "slots": [
                    {
                        "slot": slot,
                        "root": str(root),
                        "shard_rank": slot % 4,
                        "http_port": ports[slot],
                        "gpu_uuid": (_gpu_uuid(0) if duplicate_uuid_at == slot else _gpu_uuid(slot)),
                        "topology_txt_sha256": provenance.file_sha256(root / "GPU_TOPOLOGY.txt"),
                    }
                    for slot, root in enumerate(roots)
                ],
            }
        )
    )
    (parent / "ATOMIC_DONE").write_text("8\n")
    return roots, reference, barrier, ports


def _verify(
    roots: list[pathlib.Path],
    reference: pathlib.Path,
    barrier: pathlib.Path,
    ports: list[int],
    *,
    scheduler_query=None,
) -> dict:
    if scheduler_query is None:

        def scheduler_query(job_id, node):
            return {
                "job_id": job_id,
                "state": "COMPLETED",
                "exit_code": "0:0",
                "node": node,
                "alloc_tres": "billing=8,cpu=128,gres/gpu=8,mem=1000G,node=1",
            }

    return atomic.verify_atomic_gpu_coverage(
        roots,
        reference_root=reference,
        expected_node=_NODE,
        expected_allocation_job=_JOB_ID,
        expected_slots=list(range(8)),
        expected_ranks=[slot % 4 for slot in range(8)],
        expected_ports=ports,
        concurrency_barrier=barrier,
        scheduler_query=scheduler_query,
    )


def test_verifies_common_allocation_with_concurrency_and_full_provenance(tmp_path: pathlib.Path) -> None:
    roots, reference, barrier, ports = _write_fixture(tmp_path)

    result = _verify(roots, reference, barrier, ports)

    assert result["status"] == "VERIFIED"
    assert result["allocation_job_id"] == _JOB_ID
    assert result["node"] == _NODE
    assert [record["slot"] for record in result["slots"]] == list(range(8))
    assert {record["job_id"] for record in result["slots"]} == {_JOB_ID}
    assert len(set(result["gpu_uuids"])) == 8
    assert len(set(result["http_ports"])) == 8
    assert len(result["barrier_sha256"]) == 64
    assert len(result["atomic_done_sha256"]) == 64
    assert len(result["input_set_sha256"]) == 64
    assert len(result["scheduler_sha256"]) == 64
    assert result["scheduler"]["alloc_tres"].endswith("node=1")
    assert result["reference"]["job_id"] == "92999"


def test_rejects_duplicate_uuid_observed_during_concurrency(tmp_path: pathlib.Path) -> None:
    roots, reference, barrier, ports = _write_fixture(tmp_path, duplicate_uuid_at=7)

    with pytest.raises(ValueError, match="eight distinct physical GPU UUIDs"):
        _verify(roots, reference, barrier, ports)


def test_rejects_non_numeric_or_missing_concurrent_step(tmp_path: pathlib.Path) -> None:
    roots, reference, barrier, ports = _write_fixture(tmp_path)
    payload = json.loads(barrier.read_text())
    payload["numeric_step_ids"][-1] = f"{_JOB_ID}.batch"
    barrier.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="eight distinct numeric steps"):
        _verify(roots, reference, barrier, ports)


def test_rejects_barrier_slot_mapping_drift(tmp_path: pathlib.Path) -> None:
    roots, reference, barrier, ports = _write_fixture(tmp_path)
    payload = json.loads(barrier.read_text())
    payload["slots"][0]["http_port"] += 1
    barrier.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="slot mapping or topology changed"):
        _verify(roots, reference, barrier, ports)


def test_rejects_source_profile_different_from_reference(tmp_path: pathlib.Path) -> None:
    roots, reference, barrier, ports = _write_fixture(tmp_path, split_profile_at=7)

    with pytest.raises(ValueError, match="source code/profile closure differs"):
        _verify(roots, reference, barrier, ports)


def test_rejects_nonexact_atomic_done_marker(tmp_path: pathlib.Path) -> None:
    roots, reference, barrier, ports = _write_fixture(tmp_path)
    (barrier.parent / "ATOMIC_DONE").write_text("8")

    with pytest.raises(ValueError, match="ATOMIC_DONE must contain exactly"):
        _verify(roots, reference, barrier, ports)


def test_rejects_scheduler_record_without_exact_eight_gpu_allocation(tmp_path: pathlib.Path) -> None:
    roots, reference, barrier, ports = _write_fixture(tmp_path)

    def wrong_scheduler(job_id: str, node: str) -> dict:
        return {
            "job_id": job_id,
            "state": "COMPLETED",
            "exit_code": "0:0",
            "node": node,
            "alloc_tres": "billing=1,cpu=16,gres/gpu=1,mem=125G,node=1",
        }

    with pytest.raises(ValueError, match="exact cpu=128,node=1,gres/gpu=8"):
        _verify(roots, reference, barrier, ports, scheduler_query=wrong_scheduler)


def test_scheduler_query_uses_exact_parent_accounting_row(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_command = None

    def fake_run(command, **kwargs):
        nonlocal observed_command
        observed_command = command
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{_JOB_ID}|COMPLETED|0:0|{_NODE}|billing=8,cpu=128,gres/gpu=8,mem=1024000M,node=1|\n",
            stderr="",
        )

    monkeypatch.setattr(atomic.subprocess, "run", fake_run)

    record = atomic._query_scheduler_record(_JOB_ID, _NODE)  # noqa: SLF001

    assert observed_command == [
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        _JOB_ID,
        "--format=JobIDRaw,State,ExitCode,NodeList,AllocTRES",
    ]
    assert record["alloc_tres"].endswith("node=1")


def test_record_barrier_requires_three_stable_live_observations(tmp_path: pathlib.Path) -> None:
    parent = (tmp_path / "record-parent").resolve()
    ports = [34_000 + slot for slot in range(8)]
    roots = []
    for slot in range(8):
        root = parent / f"lift_bottle_{_JOB_ID}_slot{slot}"
        root.mkdir(parents=True)
        (root / "GPU_TOPOLOGY.txt").write_text(f"{slot % 4} {_gpu_uuid(slot)} {ports[slot]}\n")
        roots.append(root)
    calls = 0

    def step_query(job_id: str) -> list[str]:
        nonlocal calls
        calls += 1
        assert job_id == _JOB_ID
        return [f"{job_id}.{slot}" for slot in range(8)]

    output = parent / "CONCURRENCY_BARRIER.json"
    result = atomic.record_concurrency_barrier(
        roots,
        expected_node=_NODE,
        expected_allocation_job=_JOB_ID,
        expected_slots=list(range(8)),
        expected_ranks=[slot % 4 for slot in range(8)],
        expected_ports=ports,
        output=output,
        timeout_seconds=2,
        observation_interval_seconds=0,
        step_query=step_query,
    )

    assert calls == 4  # Three stable samples plus the atomic pre-commit recheck.
    assert result["status"] == "VERIFIED_CONCURRENT"
    assert json.loads(output.read_text()) == result
    assert output.read_bytes().endswith(b"\n")


def test_record_barrier_rejects_done_before_concurrency(tmp_path: pathlib.Path) -> None:
    parent = (tmp_path / "record-parent").resolve()
    ports = [35_000 + slot for slot in range(8)]
    roots = []
    for slot in range(8):
        root = parent / f"lift_bottle_{_JOB_ID}_slot{slot}"
        root.mkdir(parents=True)
        (root / "GPU_TOPOLOGY.txt").write_text(f"{slot % 4} {_gpu_uuid(slot)} {ports[slot]}\n")
        roots.append(root)
    (roots[0] / "DONE").write_text("1\n")

    with pytest.raises(ValueError, match="committed DONE before"):
        atomic.record_concurrency_barrier(
            roots,
            expected_node=_NODE,
            expected_allocation_job=_JOB_ID,
            expected_slots=list(range(8)),
            expected_ranks=[slot % 4 for slot in range(8)],
            expected_ports=ports,
            output=parent / "CONCURRENCY_BARRIER.json",
            timeout_seconds=1,
            observation_interval_seconds=0,
            step_query=lambda job_id: [f"{job_id}.{slot}" for slot in range(8)],
        )


def test_verify_cli_writes_atomic_output(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "nested/coverage.json"
    expected = {"schema_version": 1, "status": "VERIFIED"}
    monkeypatch.setattr(atomic, "verify_atomic_gpu_coverage", lambda *args, **kwargs: expected)
    argv = [
        "verify-atomic",
        "verify",
        "--reference-root",
        "/fixture/reference",
        "--expected-node",
        _NODE,
        "--expected-allocation-job",
        _JOB_ID,
        "--concurrency-barrier",
        "/fixture/CONCURRENCY_BARRIER.json",
        "--output",
        str(output),
    ]
    for slot in range(8):
        argv.extend(
            [
                "--root",
                f"/fixture/lift_bottle_{_JOB_ID}_slot{slot}",
                "--expected-slot",
                str(slot),
                "--expected-rank",
                str(slot % 4),
                "--expected-port",
                str(36_000 + slot),
            ]
        )
    monkeypatch.setattr(sys, "argv", argv)

    atomic.main()

    assert json.loads(output.read_text()) == expected
    assert output.read_bytes().endswith(b"\n")
