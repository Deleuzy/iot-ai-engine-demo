from __future__ import annotations

from pathlib import Path

import pytest

from iot_ai.data import ChecksumError, _fetch_one, sha256_of


def test_checksum_mismatch_is_refused(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text("hello", encoding="utf-8")
    dest = tmp_path / "raw" / "src.txt"
    with pytest.raises(ChecksumError):
        _fetch_one(src.as_uri(), "0" * 64, dest)
    assert not dest.exists()


def test_checksum_match_is_kept(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text("hello", encoding="utf-8")
    expected = sha256_of(src)
    dest = tmp_path / "raw" / "src.txt"
    assert _fetch_one(src.as_uri(), expected, dest) == dest
    assert sha256_of(dest) == expected
