"""Anomaly detectors and their evaluation on the held-out time split.

Detectors (all flag one 10-minute bin at a time):
* rule_based: the explicit rules in validate.py (offline spike rule included, so it is not causal);
* rolling_z: causal z-score on the past 6 hours; threshold tuned on the validation slice;
* isolation_forest: unsupervised on training features, no labels used;
* gbm: gradient-boosted trees trained on the labelled training faults; threshold tuned on the validation slice.

Labels are the injected faults (answer key). Real readings may contain unlabelled faults too, so precision
here is an estimate of what the injected faults show, not a measure of real-world precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from iot_ai.features import FEATURE_COLUMNS

VALIDATION_FRACTION = 0.15  # last share of the training part, used only for threshold choice


@dataclass(frozen=True)
class Scores:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int


def score(y_true: np.ndarray, y_pred: np.ndarray) -> Scores:
    t = y_true.astype(bool)
    p = y_pred.astype(bool)
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    tn = int((~t & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Scores(round(precision, 4), round(recall, 4), round(f1, 4), tp, fp, fn, tn)


def split_masks(frame: pd.DataFrame, cut: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks for (train-fit, validation, test) by time. Validation is the last part before the cut."""
    times = frame["bin"].to_numpy()
    train_all = times < np.datetime64(cut)
    train_times = np.sort(times[train_all])
    val_start = train_times[int(len(train_times) * (1 - VALIDATION_FRACTION))]
    fit = train_all & (times < np.datetime64(val_start))
    val = train_all & (times >= np.datetime64(val_start))
    test = times >= np.datetime64(cut)
    return fit, val, test


def _tune_threshold(y: np.ndarray, values: np.ndarray, candidates: list[float]) -> float:
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        f1 = score(y, values > t).f1
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def run_detectors(
    frame: pd.DataFrame, rule_flags: pd.DataFrame, cut: pd.Timestamp, seed: int, unlabelled_fault: np.ndarray
) -> tuple[dict[str, pd.Series], dict[str, float], pd.Series]:
    """0/1 prediction per detector aligned with `frame`, the thresholds chosen on the validation slice, and GBM scores."""
    fit, val, _ = split_masks(frame, cut)
    # Models are fitted without bins where the rules see a real fault that is not an injected label:
    # those would teach the model that bad real readings are normal. The test split is not changed.
    fit_clean = fit & ~unlabelled_fault
    y = frame["label"].to_numpy()
    feats = frame[FEATURE_COLUMNS].copy()
    preds: dict[str, pd.Series] = {}

    rules = rule_flags["any_rule"].to_numpy().astype(int)
    preds["rule_based"] = pd.Series(rules, index=frame.index)
    # Ablation: the same rules without "missing", since 1 in 3 bins has no reading in the real data.
    no_missing = rule_flags[["r_range", "r_stuck", "r_spike"]].any(axis=1).to_numpy().astype(int)
    preds["rule_based_no_missing"] = pd.Series(no_missing, index=frame.index)

    z = frame[["z_temp_36", "z_hum_36", "z_volt_36"]].abs().max(axis=1).fillna(0.0).to_numpy()
    k = _tune_threshold(y[val], z[val], [2.0, 3.0, 4.0, 5.0, 6.0])
    preds["rolling_z"] = pd.Series((z > k).astype(int), index=frame.index)

    filled = feats.fillna(feats[fit_clean].median()).fillna(0.0).to_numpy()
    forest = IsolationForest(n_estimators=200, contamination=0.02, random_state=seed)
    forest.fit(filled[fit_clean])
    preds["isolation_forest"] = pd.Series((forest.predict(filled) == -1).astype(int), index=frame.index)

    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=seed)
    model.fit(feats[fit_clean].to_numpy(), y[fit_clean])
    proba = model.predict_proba(feats.to_numpy())[:, 1]
    thr = _tune_threshold(y[val], proba[val], [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    preds["gbm"] = pd.Series((proba > thr).astype(int), index=frame.index)
    proba_series = pd.Series(proba, index=frame.index)
    thresholds = {"rolling_z_k": k, "gbm_threshold": thr}
    return preds, thresholds, proba_series


def evaluate(
    frame: pd.DataFrame, preds: dict[str, pd.Series], test: np.ndarray, unlabelled_fault: np.ndarray
) -> dict[str, dict[str, object]]:
    """Scores on the test split: overall and per device (median over devices that have at least one fault)."""
    y = frame["label"].to_numpy()
    motes = frame["mote"].to_numpy()
    out: dict[str, dict[str, object]] = {}
    for name, pred in preds.items():
        p = pred.to_numpy()
        overall = score(y[test], p[test])
        per_dev: list[float] = []
        for mote in np.unique(motes):
            m = test & (motes == mote)
            if y[m].sum() == 0:
                continue
            per_dev.append(score(y[m], p[m]).f1)
        by_kind: dict[str, float] = {}
        kinds = frame["fault_kind"].to_numpy()
        for kind in ["stuck", "spike", "dropout", "sag"]:
            km = test & (kinds == kind)
            by_kind[kind] = round(float(p[km].mean()), 4) if km.any() else 0.0
        # Flags on bins where the rules see a real fault that is not labelled. They count as false positives in
        # `precision`. The adjusted number removes them. It is circular for the rule detectors, which flag exactly
        # these bins, so read it only for the ML detectors.
        fp_real = int(((p == 1) & (y == 0) & unlabelled_fault & test).sum())
        adj_fp = max(overall.fp - fp_real, 0)
        adj_precision = overall.tp / (overall.tp + adj_fp) if overall.tp + adj_fp else 0.0
        out[name] = {
            "recall_by_fault_kind": by_kind,
            "flags_on_unlabelled_real_faults": fp_real,
            "precision_excluding_unlabelled_real_faults": round(adj_precision, 4),
            "precision": overall.precision,
            "recall": overall.recall,
            "f1": overall.f1,
            "tp": overall.tp,
            "fp": overall.fp,
            "fn": overall.fn,
            "tn": overall.tn,
            "devices_with_faults": len(per_dev),
            "per_device_f1_median": round(float(np.median(per_dev)), 4) if per_dev else 0.0,
        }
    return out
