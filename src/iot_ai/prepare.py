"""Build the per-device 10-minute grid, make the time split, and inject labelled faults (the answer key)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from iot_ai.config import BIN_MINUTES, WINDOW_END, WINDOW_START, InjectionPlan, Settings

CHANNELS = ["temperature", "humidity", "light", "voltage"]


def clean_readings(raw: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Keep readings inside the window with a timestamp and a device id. Clock-error rows are reported elsewhere."""
    ts = raw["timestamp"]
    keep = ts.notna() & (ts >= pd.Timestamp(WINDOW_START)) & (ts < pd.Timestamp(WINDOW_END))
    return raw.loc[keep, ["mote", "timestamp", *CHANNELS]].copy()


def build_grid(readings: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """One row per device per 10-minute bin over the whole window. Bins without readings stay NaN.

    The bin value is the median of the readings in it, so a single bad reading does not move it much.
    Out-of-range values are kept, so that the range rule can see them.
    """
    df = readings.copy()
    df["bin"] = df["timestamp"].dt.floor(f"{BIN_MINUTES}min")
    agg = df.groupby(["mote", "bin"])[CHANNELS].median().reset_index()
    bins = pd.date_range(WINDOW_START, WINDOW_END, freq=f"{BIN_MINUTES}min", inclusive="left")
    motes = sorted(int(m) for m in agg["mote"].unique())
    full = pd.MultiIndex.from_product([motes, bins], names=["mote", "bin"])
    grid = agg.set_index(["mote", "bin"]).reindex(full).reset_index()
    counts = df.groupby(["mote", "bin"]).size().rename("n_readings").reset_index()
    grid = grid.merge(counts, on=["mote", "bin"], how="left")
    grid["n_readings"] = grid["n_readings"].fillna(0).astype("int64")
    return grid.sort_values(["mote", "bin"]).reset_index(drop=True)


def select_devices(grid: pd.DataFrame, min_readings: int) -> pd.DataFrame:
    """Keep devices with enough readings and at least one temperature value and one voltage value."""
    totals = grid.groupby("mote")["n_readings"].sum()
    ok = totals[totals >= min_readings].index
    usable = grid[grid["mote"].isin(ok)]
    has_values = usable.groupby("mote")[["temperature", "voltage"]].count().min(axis=1) > 0
    return usable[usable["mote"].isin(has_values[has_values].index)].reset_index(drop=True)


def split_bin(grid: pd.DataFrame, fraction: float) -> pd.Timestamp:
    """One cut-off time shared by all devices: the first `fraction` of the window is training data."""
    bins = pd.Series(grid["bin"].unique()).sort_values().reset_index(drop=True)
    return pd.Timestamp(bins.iloc[int(len(bins) * fraction)])


@dataclass(frozen=True)
class InjectedEvent:
    mote: int
    kind: str  # stuck | spike | dropout | sag
    split: str  # train | test
    start: pd.Timestamp
    end: pd.Timestamp
    bins: int


def _free_starts(ok: np.ndarray, used: np.ndarray, span: int) -> np.ndarray:
    """Start positions where `span` bins all have data and no used bin lies within one bin of them."""
    n = len(ok)
    if n < span:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, n - span + 1)
    ok_cs = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    all_ok = (ok_cs[starts + span] - ok_cs[starts]) == span
    padded = np.concatenate([[0], used.astype(np.int64), [0]])  # padded[k] is used[k-1]
    pad_cs = np.concatenate([[0], np.cumsum(padded)])
    # padded[s : s+span+2] covers the event bins and one bin on each side.
    free = (pad_cs[starts + span + 2] - pad_cs[starts]) == 0
    return np.asarray(starts[all_ok & free], dtype=np.int64)


def inject(
    grid: pd.DataFrame, cut: pd.Timestamp, plan: InjectionPlan, seed: int
) -> tuple[pd.DataFrame, list[InjectedEvent]]:
    """Return a copy of the grid with faults written into the values, a 0/1 label column, and the event list.

    Faults are written into the measured values as a faulty device would show them:
    stuck = temperature frozen at its value at the start; spike = temperature offset for a few bins;
    dropout = all channels missing; sag = voltage ramps down (battery or power fault).
    Labels are per bin, 1 when the bin lies inside an injected event.
    """
    rng = np.random.default_rng(seed)
    out = grid.copy()
    out["label"] = 0
    out["fault_kind"] = ""
    events: list[InjectedEvent] = []
    plan_items: list[tuple[str, int, int]] = [
        ("stuck", plan.stuck_events, plan.stuck_bins),
        ("spike", plan.spike_events, plan.spike_bins),
        ("dropout", plan.dropout_events, plan.dropout_bins),
        ("sag", plan.sag_events, plan.sag_bins),
    ]
    for mote_key, dev in out.groupby("mote", sort=True):
        mote = int(str(mote_key))
        idx_all = dev.index.to_numpy()
        bins = dev["bin"].to_numpy()
        # A fault can only be written where the channel it changes has a value, so each label marks a visible change.
        # A dropout needs readings to remove, because real silent bins have no readings at all.
        valid_for = {
            "stuck": dev["temperature"].notna().to_numpy(),
            "spike": dev["temperature"].notna().to_numpy(),
            "sag": dev["voltage"].notna().to_numpy(),
            "dropout": (dev["n_readings"] > 0).to_numpy(),
        }
        for split_name, mask in (("train", bins < np.datetime64(cut)), ("test", bins >= np.datetime64(cut))):
            pos = np.flatnonzero(mask)
            used = np.zeros(len(pos), dtype=bool)
            # Kinds are shuffled. Each event gets a random start where its bins have data and no other event lies
            # within one bin, so faults inside one split never overlap or touch each other.
            order = [(kind, span) for kind, n, span in plan_items for _ in range(n)]
            order = [order[i] for i in rng.permutation(len(order))]
            for kind, span in order:
                ok = valid_for[kind][pos]
                starts = _free_starts(ok, used, span)
                if starts.size == 0:
                    # Some devices have no data at all in a split (they went quiet). Nothing can be placed there,
                    # so the event is skipped and the placed counts in detection_metrics.json show the shortfall.
                    continue
                start = int(starts[rng.integers(0, starts.size)])
                used[start : start + span] = True
                rows = idx_all[pos[start : start + span]]
                if kind == "stuck":
                    out.loc[rows, "temperature"] = float(out.loc[rows[0], "temperature"])
                elif kind == "spike":
                    out.loc[rows, "temperature"] = out.loc[rows, "temperature"] + plan.spike_temp_add
                elif kind == "dropout":
                    out.loc[rows, CHANNELS] = np.nan
                    out.loc[rows, "n_readings"] = 0
                else:
                    ramp = np.linspace(0.0, -plan.sag_volt_drop, span)
                    out.loc[rows, "voltage"] = out.loc[rows, "voltage"] + ramp
                out.loc[rows, "label"] = 1
                out.loc[rows, "fault_kind"] = kind
                events.append(
                    InjectedEvent(
                        mote=mote,
                        kind=kind,
                        split=split_name,
                        start=pd.Timestamp(bins[pos[start]]),
                        end=pd.Timestamp(bins[pos[start + span - 1]]),
                        bins=span,
                    )
                )
    return out, events
