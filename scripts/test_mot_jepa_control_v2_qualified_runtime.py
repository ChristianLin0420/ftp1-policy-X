from __future__ import annotations

import pytest

from scripts import mot_jepa_control_v2_qualified_runtime as qualified


def test_file_and_tree_verification_are_fail_closed(tmp_path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    first = root / "a.bin"
    second = root / "b.bin"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    rows = [(f"fixture/{path.name}", qualified.file_sha256(path), path.stat().st_size) for path in (first, second)]
    expected_tree = (2, 6, qualified.rows_digest(rows))

    qualified.verify_file(first, (3, qualified.file_sha256(first)))
    qualified.verify_tree(root, expected_tree, prefix="fixture")

    first.write_bytes(b"eno")
    with pytest.raises(ValueError, match="artifact mismatch"):
        qualified.verify_file(first, (3, qualified.file_sha256(second)))
    with pytest.raises(ValueError, match="tree mismatch"):
        qualified.verify_tree(root, expected_tree, prefix="fixture")
