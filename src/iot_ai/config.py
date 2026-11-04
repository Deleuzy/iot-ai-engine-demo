"""Fixed settings: where the public data comes from, its checksum, and every rule threshold."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
KNOWLEDGE_DIR = ROOT / "knowledge"

# Intel Berkeley Research Lab sensor data (54 Mica2Dot motes, 28 Feb to 5 Apr 2004, about 2.3 million
# readings of temperature, humidity, light and battery voltage). Public, freely downloadable.
# Source page: http://db.csail.mit.edu/labdata/labdata.html
DATA_URL = "https://db.csail.mit.edu/labdata/data.txt.gz"
DATA_SHA256 = "d99288c8f406ca6604d359ceaa0d8adfffa79e7095061a1e27dc4399f48c7225"
LOCS_URL = "https://db.csail.mit.edu/labdata/mote_locs.txt"
LOCS_SHA256 = "3865c0263110c24c40e3377690cecaa552e0575cf56cdb9f5f8bd17130b6bf04"

BIN_MINUTES = 10
SEED = 20260401
TRAIN_FRACTION = 0.7
# Readings outside this window are clock errors in the raw file (timestamps from other years or dates).
WINDOW_START = "2004-02-28"
WINDOW_END = "2004-04-07"


@dataclass(frozen=True)
class Rules:
    """Explicit validation rules. Every rule has a fixed threshold; none is learned."""

    temp_range: tuple[float, float] = (-10.0, 50.0)
    humidity_range: tuple[float, float] = (0.0, 100.0)
    voltage_range: tuple[float, float] = (1.8, 3.3)
    stuck_min_bins: int = 6  # identical temperature for one hour on the 10-minute grid
    spike_temp_c: float = 5.0  # jump away from both neighbours
    spike_humidity_pct: float = 15.0
    clock_gap_minutes: float = 60.0  # gap between consecutive raw readings of one mote
    min_readings_per_mote: int = 1000


@dataclass(frozen=True)
class InjectionPlan:
    """How many faults of each kind are injected into each split of each device (the answer key)."""

    stuck_events: int = 3
    stuck_bins: int = 18
    spike_events: int = 4
    spike_bins: int = 2
    spike_temp_add: float = 8.0
    dropout_events: int = 3
    dropout_bins: int = 12
    sag_events: int = 2
    sag_bins: int = 36
    sag_volt_drop: float = 0.3


@dataclass(frozen=True)
class Settings:
    rules: Rules = field(default_factory=Rules)
    injection: InjectionPlan = field(default_factory=InjectionPlan)
    seed: int = SEED
    train_fraction: float = TRAIN_FRACTION
