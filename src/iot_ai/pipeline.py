"""End-to-end run: fetch, validate, build the grid, inject faults, detect, forecast, write results.

Everything written to results/ comes from this function. The README numbers are copied from those files.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from iot_ai.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR, Settings
from iot_ai.data import checksum_report, fetch, load_raw
from iot_ai.detect import evaluate, run_detectors, split_masks
from iot_ai.features import add_features
from iot_ai.forecast import run_forecast
from iot_ai.prepare import build_grid, clean_readings, inject, select_devices, split_bin
from iot_ai.validate import audit_raw, detect_on_grid

EVENTS_KEPT = 300


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def group_events(
    frame: pd.DataFrame, flagged: pd.Series, score: pd.Series, rule_flags: pd.DataFrame
) -> list[dict[str, Any]]:
    """Consecutive flagged bins of one device become one event. Rule names come from the rules that fired inside it."""
    rows = frame[["mote", "bin", "label", "fault_kind"]].copy()
    rows["flag"] = flagged.to_numpy().astype(bool)
    rows["score"] = score.to_numpy()
    for col in ["r_missing", "r_range", "r_stuck", "r_spike"]:
        rows[col] = rule_flags[col].to_numpy().astype(bool)
    rows = rows[rows["flag"]].sort_values(["mote", "bin"])
    if rows.empty:
        return []
    new_run = (rows["mote"] != rows["mote"].shift(1)) | (rows["bin"] - rows["bin"].shift(1) != pd.Timedelta("10min"))
    rows["event_id"] = new_run.cumsum()
    events: list[dict[str, Any]] = []
    for _, ev in rows.groupby("event_id"):
        rules = [c[2:] for c in ["r_missing", "r_range", "r_stuck", "r_spike"] if bool(ev[c].any())]
        events.append(
            {
                "device": int(ev["mote"].iloc[0]),
                "start": str(ev["bin"].iloc[0]),
                "end": str(ev["bin"].iloc[-1]),
                "bins": int(len(ev)),
                "max_score": round(float(ev["score"].max()), 4),
                "rules_fired": rules,
                "injected_overlap": bool(ev["label"].max() == 1),
                "injected_kinds": sorted({k for k in ev["fault_kind"].tolist() if k}),
            }
        )
    events.sort(key=lambda e: (-float(e["max_score"]), str(e["start"])))
    return events


def run(
    settings: Settings | None = None,
    raw_dir: Path = RAW_DIR,
    results_dir: Path = RESULTS_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> dict[str, Any]:
    settings = settings or Settings()
    data_path, locs_path = fetch(raw_dir)
    raw, parse_counts = load_raw(data_path)
    raw_audit = audit_raw(raw, settings.rules)
    _write_json(
        results_dir / "validation_raw.json",
        {"file_checksums_sha256": checksum_report([data_path, locs_path]), "parse": parse_counts, "rules": raw_audit},
    )

    readings = clean_readings(raw, settings)
    grid = select_devices(build_grid(readings, settings), settings.rules.min_readings_per_mote)
    cut = split_bin(grid, settings.train_fraction)

    real_rules = detect_on_grid(grid, settings.rules)
    real_counts = {
        col: {
            "train": int(real_rules.loc[real_rules["bin"] < cut, col].sum()),
            "test": int(real_rules.loc[real_rules["bin"] >= cut, col].sum()),
        }
        for col in ["r_missing", "r_range", "r_stuck", "r_spike"]
    }
    _write_json(
        results_dir / "validation_grid_real.json",
        {
            "devices": int(grid["mote"].nunique()),
            "bins_per_device": int(grid[grid["mote"] == grid["mote"].iloc[0]].shape[0]),
            "split_cut": str(cut),
            "bins_with_no_reading": int((grid["n_readings"] == 0).sum()),
            "rule_hits_unlabelled_real_readings": real_counts,
        },
    )

    injected, events = inject(grid, cut, settings.injection, settings.seed)
    frame = add_features(injected)
    rule_flags = detect_on_grid(injected, settings.rules)
    fit, _val, test = split_masks(frame, cut)
    real_fault = real_rules[["r_range", "r_stuck", "r_spike"]].any(axis=1).to_numpy() & (frame["label"].to_numpy() == 0)
    preds, thresholds, gbm_score = run_detectors(frame, rule_flags, cut, settings.seed, real_fault)
    metrics = evaluate(frame, preds, test, real_fault)
    forecast = run_forecast(frame, test, fit, settings.seed)

    kinds: dict[str, dict[str, int]] = {}
    for ev in events:
        kinds.setdefault(ev.split, {}).setdefault(ev.kind, 0)
        kinds[ev.split][ev.kind] += 1
    _write_json(
        results_dir / "detection_metrics.json",
        {
            "seed": settings.seed,
            "split_cut": str(cut),
            "test_bins": int(test.sum()),
            "test_positive_bins": int(frame.loc[test, "label"].sum()),
            "injected_events": kinds,
            "settings": {
                "rules": asdict(settings.rules),
                "injection": asdict(settings.injection),
            },
            "thresholds": thresholds,
            "detectors": metrics,
        },
    )
    _write_json(results_dir / "forecast_metrics.json", forecast)

    processed_dir.mkdir(parents=True, exist_ok=True)
    gbm_flags = preds["gbm"]
    test_events = group_events(
        frame[test].reset_index(drop=True),
        gbm_flags[test].reset_index(drop=True),
        gbm_score[test].reset_index(drop=True),
        rule_flags[test].reset_index(drop=True),
    )
    _write_json(processed_dir / "events.json", test_events[:EVENTS_KEPT])
    # The label and fault_kind columns are kept for evaluation only; the AI engine never reads them.
    injected_out = injected[
        ["mote", "bin", "temperature", "humidity", "light", "voltage", "n_readings", "label", "fault_kind"]
    ]
    injected_out.to_csv(processed_dir / "grid.csv.gz", index=False, compression="gzip")
    _write_json(processed_dir / "injected_events.json", [asdict(e) for e in events])
    return {"devices": int(grid["mote"].nunique()), "test_events": len(test_events), "metrics": metrics}
