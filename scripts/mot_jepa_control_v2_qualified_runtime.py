#!/usr/bin/env python
"""Verify the byte-identical UniVTAC runtime and official FTP-1 comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any

CONTAINER = (
    38_711_164_928,
    "d8a83ddb9cf71fa37f4a39cc84a3244ec3a7f3c44419128a14eec29e39835af0",
)
EXTERNAL_ASSET_TREE = (
    237,
    429_248_620,
    "534a909c5c09a21878d0569e52bd0fcf5137b6662511bb51ff9f38c9ae7af368",
)
LOCAL_ASSET_TREE = (
    65,
    58_445_582,
    "66be4cda23ac6b11728186103f0b856273e2778f1f61a5bc65fa4a4ff57c1e67",
)
FTP1_OVERLAY_TREE = (
    9_459,
    1_064_700_713,
    "c29aa0062293776a44fe584feed0b8ccf34be6afebf702241f87a724c69448ea",
)
FTP1_FILES = {
    "hpt_tokenizer/GelSightMini_image_224_224_3.safetensors": (
        88_029_208,
        "657955e80a6888d0dd7a5731a5a3a279dcf65157fac165e82fa320f43b94c207",
    ),
    "hpt_tokenizer/shared_image_chunk_encoder.safetensors": (
        256_757_104,
        "9453d7b49b9c3ff65858629c9d375aa37516801ad954fe2d28a2efd7794ab29e",
    ),
    "metadata.pt": (6_131, "250240d490567c63b6ddc64d23e57cc3fe7e2c81253ea0bdd5e9e0c78e5ca658"),
    "model.safetensors": (
        7_904_019_864,
        "87534dd13263481b12daa303c12291cf2e72e07266f7d3056aad58f9426fe35a",
    ),
    "model_config.json": (851, "157393b0680bcbf3d8da9c145c93178cc72beb0ad727ec0c9e316223cceace1a"),
    "normalization/UniVTAC_lift_bottle/contact_detection_thresholds.json": (
        1_868,
        "381ea08fe779444032e3f1a7650ef40b98256ee9089108237031e529be540d99",
    ),
    "normalization/UniVTAC_lift_bottle/independent_norm_stats_all_t0_zscore.json": (
        10_056,
        "a109951bcaf284fe874ccb9f964602775a313229c7a033d48a9666e6f3e0ca4e",
    ),
    "normalization/UniVTAC_lift_bottle/train_val_split.json": (
        548,
        "1f66069eaf37f0b23d43e166edc8667bd3c68a0c2c4f5811e8cf03a0fcf2c8ad",
    ),
    "normalization/action_group_frequency_stats_train.json": (
        2_245,
        "8e1d66f57d4ec75b369de09a410bdcef4e74e6a6be023088ba2173b0b2a6c858",
    ),
    "normalization/dataset_stats.json": (
        6_343,
        "ddfe26b2c5ec53027dea13402426e7a3bc8451839821b0c40a892ab245991d2a",
    ),
    "normalization/norm_params_snapshot.json": (
        710,
        "23c775b63aebf20bea73718acca77dc23c0dd5192b606871fa1ee6bcc4dce99a",
    ),
    "normalization/share_norm_stats_all_t0_zscore.json": (
        106,
        "9d8893778ea1e4928b0c62f4af1aaf4d26875762d83efa68c2361da5493f8f04",
    ),
    "tactile_input_config_file.json": (
        266,
        "08c3ca3b6a75765e16a05a2ebefa91b324a4e68abdc73191d99823b9ed2223a9",
    ),
    "train_config.json": (
        2_503,
        "7acd19a540397c390a1c1c332b33280e1c1aefa6700d6dbe99a7587ee405bed8",
    ),
}
READY = (907, "135092bfbd21ef4db2f4fc71b81669acbf7cbe4d729a4abd3d9963db04e82c8b")
TOKENIZER = (4_264_023, "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6")


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: pathlib.Path, expected: tuple[int, str]) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    actual = (stat.st_size, file_sha256(path))
    if actual != expected:
        raise ValueError(f"qualified artifact mismatch {path}: actual={actual}, expected={expected}")
    return {"path": str(path), "bytes": actual[0], "sha256": actual[1]}


def rows_digest(rows: list[tuple[str, str, int]]) -> str:
    payload = "".join(f"{digest}\t{size}\t{logical}\n" for logical, digest, size in sorted(rows)).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_tree(
    root: pathlib.Path,
    expected: tuple[int, int, str],
    *,
    prefix: str,
    files: list[pathlib.Path] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    selected = files if files is not None else sorted(path for path in root.rglob("*") if path.is_file())
    rows = []
    for path in sorted(selected):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        stat = path.stat()
        logical = f"{prefix}/{path.resolve().relative_to(root).as_posix()}"
        rows.append((logical, file_sha256(path), stat.st_size))
    actual = (len(rows), sum(row[2] for row in rows), rows_digest(rows))
    if actual != expected:
        raise ValueError(f"qualified tree mismatch {root}: actual={actual}, expected={expected}")
    return {"path": str(root), "file_count": actual[0], "bytes": actual[1], "sha256": actual[2]}


def verify_common_runtime(*, image: pathlib.Path, assets: pathlib.Path, repo_root: pathlib.Path) -> dict[str, Any]:
    univtac = repo_root.resolve() / "UniVTAC"
    local_files = [
        path
        for path in (univtac / "assets").rglob("*")
        if path.is_file() and not ({".thumbs", ".cache", "__pycache__"} & set(path.parts))
    ]
    local_files += [univtac / "task_config/demo.yml", univtac / "policy/task_settings.json"]
    return {
        "container": verify_file(image, CONTAINER),
        "external_assets": verify_tree(assets, EXTERNAL_ASSET_TREE, prefix="tacex-assets"),
        "local_assets": verify_tree(univtac, LOCAL_ASSET_TREE, prefix="UniVTAC", files=local_files),
    }


def verify_official_runtime(
    *,
    overlay: pathlib.Path,
    checkpoint: pathlib.Path,
    openpi_data_home: pathlib.Path,
) -> dict[str, Any]:
    files = {relative: verify_file(checkpoint / relative, expected) for relative, expected in FTP1_FILES.items()}
    tokenizer = openpi_data_home / "big_vision/paligemma_tokenizer.model"
    return {
        "checkpoint": files,
        "overlay": verify_tree(overlay, FTP1_OVERLAY_TREE, prefix="ftp1-overlay"),
        "ready": verify_file(overlay / "READY.json", READY),
        "tokenizer": verify_file(tokenizer, TOKENIZER),
    }


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(raw)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=pathlib.Path)
    parser.add_argument("--assets", required=True, type=pathlib.Path)
    parser.add_argument("--repo-root", required=True, type=pathlib.Path)
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--overlay", type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--openpi-data-home", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "qualification": "univtac_ftp1_official_v1" if args.official else "univtac_common_runtime_v1",
        "common": verify_common_runtime(image=args.image, assets=args.assets, repo_root=args.repo_root),
    }
    if args.official:
        if args.overlay is None or args.checkpoint is None or args.openpi_data_home is None:
            parser.error("--official requires --overlay, --checkpoint, and --openpi-data-home")
        payload["official"] = verify_official_runtime(
            overlay=args.overlay,
            checkpoint=args.checkpoint,
            openpi_data_home=args.openpi_data_home,
        )
    elif any(value is not None for value in (args.overlay, args.checkpoint, args.openpi_data_home)):
        parser.error("official-only paths require --official")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["qualification_sha256"] = hashlib.sha256(canonical).hexdigest()
    _atomic_json(args.output, payload)
    print(json.dumps({"qualification": payload["qualification"], "sha256": payload["qualification_sha256"]}))


if __name__ == "__main__":
    main()
