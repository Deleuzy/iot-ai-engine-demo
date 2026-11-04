from __future__ import annotations

import numpy as np
import pandas as pd

from iot_ai.config import InjectionPlan
from iot_ai.features import FEATURE_COLUMNS, add_features
from iot_ai.prepare import inject


def test_features_are_causal(grid: pd.DataFrame) -> None:
    """Changing a value in the future must not change any feature at an earlier time."""
    base = add_features(grid)
    changed = grid.copy()
    mask = (changed["mote"] == 1) & (changed["bin"] >= pd.Timestamp("2004-03-02"))
    changed.loc[mask, "temperature"] = 99.0
    changed.loc[mask, "voltage"] = 1.0
    other = add_features(changed)
    before = base["bin"] < pd.Timestamp("2004-03-02")
    for col in FEATURE_COLUMNS:
        a = base.loc[before, col].to_numpy(dtype=float)
        b = other.loc[before, col].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True), col


def test_injection_is_non_overlapping_and_labelled(grid: pd.DataFrame) -> None:
    cut = pd.Timestamp("2004-03-02 12:00")
    plan = InjectionPlan(
        stuck_events=1,
        stuck_bins=6,
        spike_events=1,
        spike_bins=2,
        dropout_events=1,
        dropout_bins=4,
        sag_events=1,
        sag_bins=10,
    )
    out, events = inject(grid, cut, plan, seed=3)
    per_split = {}
    for split in ("train", "test"):
        per_split[split] = [e for e in events if e.split == split]
        assert len(per_split[split]) == 8  # 4 kinds x 2 devices
    total_bins = sum(e.bins for e in events)
    assert int(out["label"].sum()) == total_bins  # no overlaps, so label bins equal event bins
    assert set(out.loc[out["label"] == 1, "fault_kind"]) <= {"stuck", "spike", "dropout", "sag"}
    assert (out.loc[out["label"] == 0, "fault_kind"] == "").all()
