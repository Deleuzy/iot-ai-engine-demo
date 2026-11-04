"""Download the public dataset with a pinned SHA-256, and parse the raw readings."""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from iot_ai.config import DATA_SHA256, DATA_URL, LOCS_SHA256, LOCS_URL, RAW_DIR

COLUMNS = ["date", "time", "epoch", "mote", "temperature", "humidity", "light", "voltage"]


class ChecksumError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_one(url: str, expected: str, dest: Path) -> Path:
    """Reuse the file if its checksum matches; otherwise download and verify. Never keep a bad file."""
    if dest.exists() and sha256_of(dest) == expected:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:  # noqa: S310 - fixed https URL
        out.write(response.read())
    actual = sha256_of(tmp)
    if actual != expected:
        tmp.unlink()
        raise ChecksumError(f"{url}: sha256 {actual} does not match the pinned {expected}")
    tmp.replace(dest)
    return dest


def fetch(raw_dir: Path = RAW_DIR) -> tuple[Path, Path]:
    data = _fetch_one(DATA_URL, DATA_SHA256, raw_dir / "data.txt.gz")
    locs = _fetch_one(LOCS_URL, LOCS_SHA256, raw_dir / "mote_locs.txt")
    return data, locs


def count_lines(path: Path) -> int:
    with gzip.open(path, "rb") as handle:
        return sum(1 for _ in handle)


def load_raw(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse the whitespace-separated file. Lines that do not have 8 fields are counted and dropped."""
    total = count_lines(path)
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        compression="gzip",
        engine="c",
        on_bad_lines="skip",
        dtype={"mote": "Int64"},
    )
    timestamp = pd.to_datetime(frame["date"] + " " + frame["time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    frame = frame.assign(timestamp=timestamp)
    for col in ["temperature", "humidity", "light", "voltage"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    frame = frame.drop(columns=["date", "time"])
    frame = frame.dropna(subset=["mote"]).copy()
    frame["mote"] = frame["mote"].astype("int64")
    counts = {
        "lines_in_file": total,
        "rows_parsed": int(len(frame)),
        "rows_bad_fields_skipped": int(total - len(frame)),
        "rows_bad_timestamp": int(timestamp.isna().sum()),
    }
    return frame.reset_index(drop=True), counts


def checksum_report(paths: list[Path]) -> dict[str, str]:
    return {p.name: sha256_of(p) for p in paths}
