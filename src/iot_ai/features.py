"""Causal features per device: every value depends only on the current and earlier bins.

No feature looks ahead, so the same code can run on a live stream. The one exception is the
offline spike rule in validate.py, which is not used as a model input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "log_light",
    "voltage",
    "missing",
    "d_temp_1",
    "d_hum_1",
    "d_volt_1",
    "d_volt_144",
    "z_temp_36",
    "z_hum_36",
    "z_volt_36",
    "stuck_run",
    "n_readings",
    "silent_bins",
    "hour_sin",
    "hour_cos",
]


def _run_length_so_far(same: pd.Series, mote: pd.Series) -> pd.Series:
    """Consecutive bins up to and including now with an identical value, counted per device (causal)."""
    run_id = (~same).astype("int64").groupby(mote).cumsum()
    return same.groupby([mote, run_id]).cumcount() + 1


def add_features(grid: pd.DataFrame) -> pd.DataFrame:
    g = grid.sort_values(["mote", "bin"]).reset_index(drop=True)
    grp = g.groupby("mote")
    out = pd.DataFrame(index=g.index)
    out["mote"] = g["mote"].to_numpy()
    out["bin"] = g["bin"].to_numpy()
    out["label"] = g["label"].to_numpy()
    out["fault_kind"] = g["fault_kind"].to_numpy()
    out["temperature"] = g["temperature"]
    out["humidity"] = g["humidity"]
    out["log_light"] = np.log1p(g["light"].clip(lower=0))
    out["voltage"] = g["voltage"]
    out["missing"] = (g[["temperature", "humidity", "voltage"]].isna().any(axis=1)).astype("float64")
    out["d_temp_1"] = g["temperature"] - grp["temperature"].shift(1)
    out["d_hum_1"] = g["humidity"] - grp["humidity"].shift(1)
    out["d_volt_1"] = g["voltage"] - grp["voltage"].shift(1)
    out["d_volt_144"] = g["voltage"] - grp["voltage"].shift(144)
    for col, name in (("temperature", "z_temp_36"), ("humidity", "z_hum_36"), ("voltage", "z_volt_36")):
        past = grp[col].shift(1)
        mean = past.groupby(g["mote"]).transform(lambda s: s.rolling(36, min_periods=6).mean())
        std = past.groupby(g["mote"]).transform(lambda s: s.rolling(36, min_periods=6).std())
        out[name] = (g[col] - mean) / std.replace(0.0, np.nan)
    same = (g["temperature"] == grp["temperature"].shift(1)) & g["temperature"].notna()
    out["stuck_run"] = _run_length_so_far(same, g["mote"]).astype("float64")
    # Connectivity: readings per bin, and how many bins in a row the device has been silent (causal).
    silent = g["n_readings"] == 0
    out["n_readings"] = g["n_readings"].astype("float64")
    out["silent_bins"] = _run_length_so_far(silent, g["mote"]).astype("float64") * silent.astype("float64")
    hour = g["bin"].dt.hour + g["bin"].dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return out
