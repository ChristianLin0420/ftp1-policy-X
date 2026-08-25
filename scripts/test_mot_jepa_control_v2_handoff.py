from __future__ import annotations

import json

import pytest

from scripts.mot_jepa_control_v2_handoff import PRODUCTION_EPISODES
from scripts.mot_jepa_control_v2_handoff import validate_production_collection_handoff


def _write_collection(tmp_path, **overrides):
    payload = {
        "schema_version": 3,
        "status": "DONE",
        "collection_mode": "production",
        "requested_episodes": PRODUCTION_EPISODES,
        "saved_episodes": PRODUCTION_EPISODES,
        "global_episodes": PRODUCTION_EPISODES,
        "episodes": [{"seed": index} for index in range(PRODUCTION_EPISODES)],
    }
    payload.update(overrides)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "DONE").write_text(f"{PRODUCTION_EPISODES}\n")
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    return payload


def test_production_handoff_accepts_only_exact_collection(tmp_path) -> None:
    expected = _write_collection(tmp_path)

    assert validate_production_collection_handoff(tmp_path) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("status", "RUNNING"),
        ("collection_mode", "smoke"),
        ("requested_episodes", PRODUCTION_EPISODES + 1),
        ("saved_episodes", PRODUCTION_EPISODES + 1),
        ("global_episodes", PRODUCTION_EPISODES + 1),
    ],
)
def test_production_handoff_rejects_contract_mismatch(tmp_path, field, value) -> None:
    _write_collection(tmp_path, **{field: value})

    with pytest.raises(ValueError, match="exact production V3 handoff"):
        validate_production_collection_handoff(tmp_path)


def test_production_handoff_rejects_overfull_done_and_episode_list(tmp_path) -> None:
    _write_collection(tmp_path)
    (tmp_path / "DONE").write_text(f"{PRODUCTION_EPISODES + 1}\n")
    with pytest.raises(ValueError, match="DONE must equal 1000"):
        validate_production_collection_handoff(tmp_path)

    _write_collection(tmp_path, episodes=[{"seed": index} for index in range(PRODUCTION_EPISODES + 1)])
    with pytest.raises(ValueError, match="exactly 1000 episode records"):
        validate_production_collection_handoff(tmp_path)
