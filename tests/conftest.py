"""Tests never download data. This fixture builds a small synthetic device grid with known faults."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iot_ai.ai.tools import DeviceStore
from iot_ai.config import Rules


def make_grid(days: int = 3) -> pd.DataFrame:
    bins = pd.date_range("2004-03-01", periods=days * 144, freq="10min")
    rng = np.random.default_rng(1)
    rows = []
    for mote in (1, 2):
        temp = 20 + np.sin(np.arange(len(bins)) / 20) + rng.normal(0, 0.05, len(bins))
        hum = 40 + rng.normal(0, 0.3, len(bins))
        volt = np.linspace(2.9, 2.85, len(bins)) + rng.normal(0, 0.001, len(bins))
        df = pd.DataFrame(
            {
                "mote": mote,
                "bin": bins,
                "temperature": temp,
                "humidity": hum,
                "light": 100.0,
                "voltage": volt,
                "n_readings": 3,
                "label": 0,
                "fault_kind": "",
            }
        )
        rows.append(df)
    grid = pd.concat(rows, ignore_index=True)
    # Device 1: stuck temperature for 2 hours and one spike.
    d1 = grid["mote"] == 1
    idx = grid.index[d1][50:62]
    grid.loc[idx, "temperature"] = float(grid.loc[idx[0], "temperature"])
    spike = grid.index[d1][100:102]
    grid.loc[spike, "temperature"] = grid.loc[spike, "temperature"] + 8.0
    # Device 2: a silent period (no readings, no values).
    d2 = grid.index[grid["mote"] == 2][200:212]
    grid.loc[d2, ["temperature", "humidity", "voltage"]] = np.nan
    grid.loc[d2, "n_readings"] = 0
    return grid


@pytest.fixture()
def grid() -> pd.DataFrame:
    return make_grid()


@pytest.fixture()
def processed_dir(tmp_path: Path, grid: pd.DataFrame) -> Path:
    out = tmp_path / "processed"
    out.mkdir()
    export = grid.drop(columns=["label", "fault_kind"]).copy()
    export.to_csv(out / "grid.csv.gz", index=False, compression="gzip")
    events = [
        {
            "device": 1,
            "start": "2004-03-01 08:20:00",
            "end": "2004-03-01 09:10:00",
            "bins": 6,
            "max_score": 0.9123,
            "rules_fired": ["stuck"],
            "injected_overlap": True,
            "injected_kinds": ["stuck"],
        },
        {
            "device": 2,
            "start": "2004-03-01 10:00:00",
            "end": "2004-03-01 11:50:00",
            "bins": 12,
            "max_score": 0.4411,
            "rules_fired": ["missing"],
            "injected_overlap": True,
            "injected_kinds": ["dropout"],
        },
        {
            "device": 2,
            "start": "2004-03-02 05:00:00",
            "end": "2004-03-02 05:10:00",
            "bins": 1,
            "max_score": 0.2,
            "rules_fired": [],
            "injected_overlap": False,
            "injected_kinds": [],
        },
    ]
    (out / "events.json").write_text(json.dumps(events), encoding="utf-8")
    return out


@pytest.fixture()
def store(processed_dir: Path) -> DeviceStore:
    return DeviceStore.load(processed_dir, Rules())
