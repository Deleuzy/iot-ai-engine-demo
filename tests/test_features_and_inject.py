from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iot_ai.config import InjectionPlan
from iot_ai.features import FEATURE_COLUMNS, add_features
from iot_ai.prepare import inject


@pytest.mark.parametrize("motes", [(1,), (2,), (1, 2)])
def test_features_are_causal(grid: pd.DataFrame, motes: tuple[int, ...]) -> None:
    """Changing any channel or reading count after a cut-off must not change any feature before it, on any device."""
    base = add_features(grid)
    changed = grid.copy()
    cut = pd.Timestamp("2004-03-02")
    mask = changed["mote"].isin(motes) & (changed["bin"] >= cut)
    changed.loc[mask, "temperature"] = 99.0
    changed.loc[mask, "humidity"] = 1.0
    changed.loc[mask, "light"] = 5000.0
    changed.loc[mask, "voltage"] = 1.0
    changed.loc[mask, "n_readings"] = 0
    other = add_features(changed)
    before = (base["bin"] < cut).to_numpy()
    assert before.any()
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


def test_injected_faults_sit_on_bins_with_data(grid: pd.DataFrame) -> None:
    """A label must mark a visible change: stuck and spike need a temperature, sag a voltage, dropout readings."""
    cut = pd.Timestamp("2004-03-02 12:00")
    plan = InjectionPlan(stuck_events=2, stuck_bins=6, spike_events=2, spike_bins=2, dropout_events=2,
                         dropout_bins=4, sag_events=2, sag_bins=10)
    out, _ = inject(grid, cut, plan, seed=5)
    labelled = out["label"] == 1
    kinds = out["fault_kind"]
    assert grid.loc[labelled & kinds.isin(["stuck", "spike"]), "temperature"].notna().all()
    assert grid.loc[labelled & (kinds == "sag"), "voltage"].notna().all()
    assert grid.loc[labelled & (kinds == "dropout"), "n_readings"].gt(0).all()
    dropped = out.loc[labelled & (kinds == "dropout")]
    assert dropped["n_readings"].eq(0).all() and dropped["temperature"].isna().all()
