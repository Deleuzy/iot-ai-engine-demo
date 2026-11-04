"""Tools the model may call. Each tool returns plain numbers and strings computed from the device data.

The engine validates every call (known name, required arguments, types) before running it. Labels and
injected-fault information are never loaded here: the assistant sees only what a device would show.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from iot_ai.ai.types import ToolSpec
from iot_ai.config import PROCESSED_DIR
from iot_ai.config import Rules as RulesConfig
from iot_ai.validate import detect_on_grid


class ToolError(ValueError):
    pass


def _num(value: float, digits: int) -> float | None:
    """A rounded number, or None when there is no valid value (JSON null, never NaN)."""
    if value != value:  # NaN
        return None
    return round(float(value), digits)


@dataclass
class DeviceStore:
    grid: pd.DataFrame  # sorted by device then bin; no label columns
    rule_flags: pd.DataFrame  # same order as grid
    events: list[dict[str, Any]]

    @classmethod
    def load(cls, processed_dir: Path = PROCESSED_DIR, rules: RulesConfig | None = None) -> DeviceStore:
        grid = pd.read_csv(processed_dir / "grid.csv.gz", parse_dates=["bin"])
        grid = grid.drop(columns=[c for c in ("label", "fault_kind") if c in grid.columns])
        grid = grid.sort_values(["mote", "bin"]).reset_index(drop=True)
        flags = detect_on_grid(grid, rules or RulesConfig())
        events = json.loads((processed_dir / "events.json").read_text(encoding="utf-8"))
        return cls(grid=grid, rule_flags=flags, events=events)

    def device_ids(self) -> list[int]:
        return sorted(int(m) for m in self.grid["mote"].unique())

    def history(self, device: int, start: str, end: str) -> dict[str, Any]:
        if device not in self.device_ids():
            raise ToolError(f"unknown device {device}")
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        if t1 < t0:
            raise ToolError("end is before start")
        mask = (self.grid["mote"] == device) & (self.grid["bin"] >= t0) & (self.grid["bin"] <= t1)
        g = self.grid[mask]
        f = self.rule_flags[mask.to_numpy()]
        if g.empty:
            raise ToolError("no bins in that window")
        # Out-of-range readings (for example 120 degrees from a broken sensor) are left out of the statistics.
        usable = g[~f["r_range"].to_numpy()]
        temp_ok = usable["temperature"].dropna()
        volt = g["voltage"].dropna()
        hours = max((g["bin"].iloc[-1] - g["bin"].iloc[0]).total_seconds() / 86400.0, 1e-9)
        slope = float(volt.iloc[-1] - volt.iloc[0]) / hours if len(volt) > 1 else 0.0
        silent = (g["n_readings"] == 0).astype(int)
        run = silent.groupby((silent != silent.shift()).cumsum()).cumsum()
        return {
            "device": device,
            "window_start": str(g["bin"].iloc[0]),
            "window_end": str(g["bin"].iloc[-1]),
            "bins": int(len(g)),
            "missing_fraction": _num(float(g["temperature"].isna().mean()), 3),
            "temperature_mean": _num(float(temp_ok.mean()) if len(temp_ok) else float("nan"), 2),
            "temperature_min": _num(float(temp_ok.min()) if len(temp_ok) else float("nan"), 2),
            "temperature_max": _num(float(temp_ok.max()) if len(temp_ok) else float("nan"), 2),
            "humidity_mean": _num(float(usable["humidity"].mean()) if len(usable) else float("nan"), 1),
            "bins_out_of_range_excluded": int(f["r_range"].sum()),
            "voltage_first": round(float(volt.iloc[0]), 3) if len(volt) else None,
            "voltage_last": round(float(volt.iloc[-1]), 3) if len(volt) else None,
            "voltage_change_per_day": _num(slope, 4),
            "longest_silence_bins": int(run.max()) if len(run) else 0,
            "bins_stuck": int(f["r_stuck"].sum()),
            "bins_spike": int(f["r_spike"].sum()),
            "bins_out_of_range": int(f["r_range"].sum()),
            "bins_missing": int(f["r_missing"].sum()),
        }

    def device_events(self, device: int) -> dict[str, Any]:
        if device not in self.device_ids():
            raise ToolError(f"unknown device {device}")
        mine = [e for e in self.events if e["device"] == device][:5]
        return {
            "device": device,
            "events": [
                {
                    "start": e["start"],
                    "end": e["end"],
                    "bins": e["bins"],
                    "max_score": e["max_score"],
                    "rules_fired": e["rules_fired"],
                }
                for e in mine
            ],
        }


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="get_device_history",
        description="Summary of one device over a time window: temperature, humidity, battery voltage and rule counts.",
        parameters={
            "type": "object",
            "properties": {
                "device": {"type": "integer"},
                "start": {"type": "string", "description": "ISO timestamp, e.g. 2004-03-20 00:00:00"},
                "end": {"type": "string", "description": "ISO timestamp"},
            },
            "required": ["device", "start", "end"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="get_device_events",
        description="The most recent flagged events for one device, with the rules that fired.",
        parameters={
            "type": "object",
            "properties": {"device": {"type": "integer"}},
            "required": ["device"],
            "additionalProperties": False,
        },
    ),
]
TOOL_NAMES = {t.name for t in TOOL_SPECS}


def _check_args(spec: ToolSpec, args: dict[str, Any]) -> None:
    props: dict[str, Any] = spec.parameters["properties"]
    for key in spec.parameters.get("required", []):
        if key not in args:
            raise ToolError(f"{spec.name}: missing argument {key}")
    for key, value in args.items():
        if key not in props:
            raise ToolError(f"{spec.name}: unexpected argument {key}")
        want = props[key]["type"]
        ok = (want == "integer" and isinstance(value, int) and not isinstance(value, bool)) or (
            want == "string" and isinstance(value, str)
        )
        if not ok:
            raise ToolError(f"{spec.name}: argument {key} must be {want}")


def run_tool(store: DeviceStore, name: str, args: dict[str, Any]) -> dict[str, Any]:
    spec = next((t for t in TOOL_SPECS if t.name == name), None)
    if spec is None:
        raise ToolError(f"unknown tool {name}")
    _check_args(spec, args)
    if name == "get_device_history":
        return store.history(args["device"], args["start"], args["end"])
    return store.device_events(args["device"])
