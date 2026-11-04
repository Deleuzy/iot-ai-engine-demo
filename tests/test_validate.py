from __future__ import annotations

import pandas as pd

from iot_ai.config import Rules
from iot_ai.validate import audit_raw, detect_on_grid


def test_stuck_run_is_flagged_only_when_long_enough(grid: pd.DataFrame) -> None:
    flags = detect_on_grid(grid, Rules())
    d1 = flags[flags["mote"] == 1].reset_index(drop=True)
    assert d1.loc[50:61, "r_stuck"].all()
    assert not d1.loc[0:20, "r_stuck"].any()


def test_spike_and_missing_rules(grid: pd.DataFrame) -> None:
    flags = detect_on_grid(grid, Rules())
    d1 = flags[flags["mote"] == 1].reset_index(drop=True)
    assert d1.loc[100:101, "r_spike"].all()
    d2 = flags[flags["mote"] == 2].reset_index(drop=True)
    assert d2.loc[200:211, "r_missing"].all()
    assert not d2.loc[0:100, "r_missing"].any()


def test_out_of_range_rule() -> None:
    grid = pd.DataFrame(
        {
            "mote": [1, 1],
            "bin": pd.to_datetime(["2004-03-01 00:00", "2004-03-01 00:10"]),
            "temperature": [122.0, 20.0],
            "humidity": [40.0, 40.0],
            "light": [1.0, 1.0],
            "voltage": [2.8, 2.8],
            "n_readings": [1, 1],
            "label": [0, 0],
            "fault_kind": ["", ""],
        }
    )
    flags = detect_on_grid(grid, Rules())
    assert flags["r_range"].tolist() == [True, False]


def test_raw_clock_errors_are_counted() -> None:
    raw = pd.DataFrame(
        {
            "mote": [1, 1, 1],
            "epoch": [1, 2, 3],
            "timestamp": pd.to_datetime(["2004-03-01 10:00", "2004-03-01 09:00", "1999-01-01 00:00"]),
            "temperature": [20.0, 20.0, 20.0],
            "humidity": [40.0, 40.0, 40.0],
            "light": [1.0, 1.0, 1.0],
            "voltage": [2.8, 2.8, 2.8],
        }
    )
    counts = audit_raw(raw, Rules())
    assert counts["rule_clock_step_back"] == 2
    assert counts["rule_clock_outside_window"] == 1
