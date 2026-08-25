#!/usr/bin/env python
"""Validate the exact production collection handed to MoT-Control V3 preparation."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from openpi.mot_jepa.control_v2_config import CONTROL_V2_PRODUCTION_EPISODES

PRODUCTION_EPISODES = CONTROL_V2_PRODUCTION_EPISODES


def validate_production_collection_handoff(collection_root: str | pathlib.Path) -> dict[str, Any]:
    """Require the one canonical, exact-1,000 production collection contract."""

    root = pathlib.Path(collection_root)
    done_path = root / "DONE"
    manifest_path = root / "manifest.json"
    if not done_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"collection is incomplete: {root}")

    done = done_path.read_text().strip()
    if done != str(PRODUCTION_EPISODES):
        raise ValueError(f"collection DONE must equal {PRODUCTION_EPISODES}, got {done!r}")

    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("collection manifest root must be an object")
    expected = {
        "schema_version": 3,
        "status": "DONE",
        "collection_mode": "production",
        "requested_episodes": PRODUCTION_EPISODES,
        "saved_episodes": PRODUCTION_EPISODES,
        "global_episodes": PRODUCTION_EPISODES,
    }
    mismatches = {
        name: {"actual": payload.get(name), "expected": value}
        for name, value in expected.items()
        if type(payload.get(name)) is not type(value) or payload.get(name) != value
    }
    if mismatches:
        raise ValueError(f"collection is not the exact production V3 handoff: {mismatches}")

    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != PRODUCTION_EPISODES:
        count = len(episodes) if isinstance(episodes, list) else None
        raise ValueError(f"collection manifest must contain exactly {PRODUCTION_EPISODES} episode records, got {count}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", required=True, type=pathlib.Path)
    args = parser.parse_args()
    payload = validate_production_collection_handoff(args.collection_root)
    print(
        json.dumps(
            {
                "collection_mode": payload["collection_mode"],
                "episodes": payload["saved_episodes"],
                "schema_version": payload["schema_version"],
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
