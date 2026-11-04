"""One-hour-ahead battery voltage forecast (6 bins). Baseline: persistence (tomorrow's value = now).

Evaluated only on clean test rows: no injected fault inside the input or target window, and both present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from iot_ai.features import FEATURE_COLUMNS

HORIZON_BINS = 6


def run_forecast(frame: pd.DataFrame, test: np.ndarray, fit: np.ndarray, seed: int) -> dict[str, float | int]:
    target = frame.groupby("mote")["voltage"].shift(-HORIZON_BINS)
    # Forward-looking window for each row: any label in [t, t+6] (computed per device).
    fwd = frame.groupby("mote")["label"].transform(
        lambda s: s[::-1].rolling(HORIZON_BINS + 1, min_periods=1).max()[::-1]
    )
    clean = (fwd.to_numpy() == 0) & frame["voltage"].notna().to_numpy() & target.notna().to_numpy()
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
