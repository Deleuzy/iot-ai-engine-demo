"""Explicit data-quality rules. Each rule is a plain boolean test with a fixed threshold from config.Rules.

Two levels:
* raw readings (about 30 s apart per device): missing values, out-of-range values, clock errors
  (timestamps outside the window, steps back in time), and gaps of silence (more than 60 minutes between readings).
  Gaps are counted and reported on their own; they are not clock errors, since a device can be silent for real;
* the 10-minute grid built from the readings: the same rules plus "stuck" (an identical value for an hour).
"""

from __future__ import annotations

import pandas as pd

from iot_ai.config import WINDOW_END, WINDOW_START, Rules

CHANNELS = ["temperature", "humidity", "light", "voltage"]
RULE_COLUMNS = ["r_missing", "r_range", "r_stuck", "r_spike", "r_clock"]


def _range_flag(values: pd.Series, bounds: tuple[float, float]) -> pd.Series:
    low, high = bounds
    return (values < low) | (values > high)


def audit_raw(raw: pd.DataFrame, rules: Rules) -> dict[str, int]:
    """Count rule hits on the raw readings. Returns counts only; the grid detector does the flagging."""
    df = raw.sort_values(["mote", "epoch"]).reset_index(drop=True)
    window_start = pd.Timestamp(WINDOW_START)
    window_end = pd.Timestamp(WINDOW_END)
    ts = df["timestamp"]
    outside = ts.isna() | (ts < window_start) | (ts >= window_end)
    prev_ts = df.groupby("mote")["timestamp"].shift(1)
    step_back = (ts < prev_ts) & ts.notna() & prev_ts.notna()
    gap_minutes = (ts - prev_ts).dt.total_seconds() / 60.0
    gap = gap_minutes > rules.clock_gap_minutes
    missing = df[CHANNELS].isna().any(axis=1)
    temp_out = _range_flag(df["temperature"], rules.temp_range)
    hum_out = _range_flag(df["humidity"], rules.humidity_range)
    volt_out = _range_flag(df["voltage"], rules.voltage_range)
    return {
        "readings": int(len(df)),
        "rule_clock_outside_window": int(outside.sum()),
        "rule_clock_step_back": int(step_back.sum()),
        "rule_clock_gap_over_threshold": int(gap.sum()),
        "rule_missing_any_channel": int(missing.sum()),
        "rule_out_of_range_temperature": int(temp_out.sum()),
        "rule_out_of_range_humidity": int(hum_out.sum()),
        "rule_out_of_range_voltage": int(volt_out.sum()),
    }


def detect_on_grid(grid: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """Boolean rule flags on the 10-minute grid. Input must be sorted by mote then bin.

    Spike uses a centred median, so it looks ahead and is an offline rule: it cannot run in real time.
    All other rules only look at the current and past bins.
    """
    g = grid.sort_values(["mote", "bin"]).reset_index(drop=True)
    by_mote = g.groupby("mote")
    temp = g["temperature"]
    hum = g["humidity"]
    out = pd.DataFrame(index=g.index)
    out["r_missing"] = temp.isna() | hum.isna() | g["voltage"].isna()
    out["r_range"] = (
        _range_flag(temp, rules.temp_range)
        | _range_flag(hum, rules.humidity_range)
        | _range_flag(g["voltage"], rules.voltage_range)
    )
    prev_temp = by_mote["temperature"].shift(1)
    same = (temp == prev_temp) & temp.notna()
    # A new run starts wherever the value differs from the bin before it (same device only).
    run_id = (~same).astype("int64").groupby(g["mote"]).cumsum()
    run_len = same.groupby([g["mote"], run_id]).transform("size")
    out["r_stuck"] = temp.notna() & (run_len >= rules.stuck_min_bins)
    # Deviation from the centred 7-bin median (about 70 minutes). A two-bin spike cannot hide inside the median,
    # which a rule that compares with only the two neighbours would miss.
    temp_med = by_mote["temperature"].transform(lambda s: s.rolling(7, center=True, min_periods=3).median())
    hum_med = by_mote["humidity"].transform(lambda s: s.rolling(7, center=True, min_periods=3).median())
    temp_spike = (temp - temp_med).abs() > rules.spike_temp_c
    hum_spike = (hum - hum_med).abs() > rules.spike_humidity_pct
    out["r_spike"] = temp_spike.fillna(False) | hum_spike.fillna(False)
    out["r_clock"] = False
    out["any_rule"] = out[["r_missing", "r_range", "r_stuck", "r_spike"]].any(axis=1)
    out["mote"] = g["mote"].to_numpy()
    out["bin"] = g["bin"].to_numpy()
    return out
