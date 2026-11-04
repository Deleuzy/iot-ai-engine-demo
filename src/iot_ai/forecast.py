"""One-hour-ahead battery voltage forecast (6 bins). Baseline: persistence (tomorrow's value = now).

Evaluated only on clean test rows: no injected fault inside the input or target window, and both present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from iot_ai.features import FEATURE_COLUMNS

HORIZON_BINS = 6
# The longest input look-back in features.py is 144 bins (d_volt_144). A row is clean only if no injected label
# lies in its whole input window [t-144, t] or in its target window [t+1, t+6].
LOOKBACK_BINS = 144


def run_forecast(frame: pd.DataFrame, test: np.ndarray, fit: np.ndarray, seed: int) -> dict[str, float | int]:
    target = frame.groupby("mote")["voltage"].shift(-HORIZON_BINS)
    span = LOOKBACK_BINS + HORIZON_BINS + 1
    # For row t: max label over [t-144, t+6], per device. A rolling window ending at t+6, shifted back to t.
    window_label = frame.groupby("mote")["label"].transform(
        lambda s: s.rolling(span, min_periods=1).max().shift(-HORIZON_BINS).fillna(1.0)
    )
    clean = (window_label.to_numpy() == 0) & frame["voltage"].notna().to_numpy() & target.notna().to_numpy()
    fit_m = fit & clean
    test_m = test & clean
    y = target.to_numpy()
    x = frame[FEATURE_COLUMNS].to_numpy()
    persistence = frame["voltage"].to_numpy()
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed)
    model.fit(x[fit_m], y[fit_m])
    pred = model.predict(x[test_m])
    actual = y[test_m]
    mae_model = float(np.mean(np.abs(pred - actual)))
    mae_persist = float(np.mean(np.abs(persistence[test_m] - actual)))
    return {
        "horizon_minutes": HORIZON_BINS * 10,
        "test_rows": int(test_m.sum()),
        "mae_persistence_volts": round(mae_persist, 4),
        "mae_gbm_volts": round(mae_model, 4),
        "gbm_beats_persistence": bool(mae_model < mae_persist),
    }
