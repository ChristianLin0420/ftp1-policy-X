from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

import pytest


def _module():
    path = pathlib.Path(__file__).parent.parent / "UniVTAC/scripts/parallel_collect_data.py"
    spec = importlib.util.spec_from_file_location("univtac_parallel_collect_data", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_control_v2_yaml_fields_are_routed_to_each_worker(tmp_path) -> None:
    module = _module()
    env = SimpleNamespace(
        tactile_sensor_type="old",
        random_texture=True,
        decimation=9,
        save_frequency=9,
        video_frequency=9,
        render_frequency=9,
        obs_data_type={},
        scene=SimpleNamespace(num_envs=7),
    )
    config = {
        "sensor_type": "gsmini",
        "random_texture": False,
        "decimation": 1,
        "save_frequency": 1,
        "video_frequency": 0,
        "render_frequency": 2,
        "observations": {"embodiment": ["joint", "command"]},
    }

    result = module.configure_worker_env(env, config, tmp_path, "3")

    assert result is env
    assert env.worker_name == "worker_2"
    assert env.save_dir == tmp_path / ".workers/worker_2"
    assert env.tactile_sensor_type == "gsmini"
    assert env.random_texture is False
    assert (env.decimation, env.save_frequency, env.video_frequency, env.render_frequency) == (1, 1, 0, 2)
    assert env.obs_data_type == config["observations"]
    assert env.scene.num_envs == 1


def test_collection_requires_every_requested_worker_to_start_and_participate() -> None:
    module = _module()
    healthy = [{"ready": True, "started_attempts": 1} for _ in range(4)]

    module.validate_worker_topology([0, 0, 0, 0], healthy, 4, 4)

    degraded = [dict(status) for status in healthy]
    degraded[2]["started_attempts"] = 0
    with pytest.raises(RuntimeError, match=r"no_attempt=.+Worker-3"):
        module.validate_worker_topology([0, 0, 0, 0], degraded, 4, 4)

    with pytest.raises(RuntimeError, match="Worker-2:exit=1"):
        module.validate_worker_topology([0, 1, 0, 0], healthy, 4, 4)


def test_collection_assigns_distinct_valid_isaac_http_ports() -> None:
    module = _module()

    assert module.worker_http_ports(28000, 4) == [28000, 28001, 28002, 28003]

    with pytest.raises(ValueError, match=r"must be within \[1024, 65535\]"):
        module.worker_http_ports(1023, 1)
    with pytest.raises(ValueError, match=r"must be within \[1024, 65535\]"):
        module.worker_http_ports(65534, 4)


def test_collection_seed_streams_stay_disjoint_after_failures() -> None:
    module = _module()
    base = 2_000_000
    streams = []
    for rank in range(4):
        seed = base + rank
        stream = []
        for _ in range(20):
            stream.append(seed)
            seed = module.advance_collection_seed(seed, 4)
        streams.append(set(stream))

    assert all(not streams[left] & streams[right] for left in range(4) for right in range(left + 1, 4))
    with pytest.raises(ValueError, match="seed_step must be positive"):
        module.advance_collection_seed(base, 0)


def test_empty_gpu_override_preserves_pyxis_device_routing() -> None:
    module = _module()

    assert module.split_devices("", 2) == [[], []]
    assert module.split_devices("0,1", 2) == [["0"], ["1"]]
