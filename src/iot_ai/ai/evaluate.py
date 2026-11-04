"""Engine evaluation on a fixed set of events from the held-out split. Writes results/engine_eval.json.

Three things are measured:
* retrieval: is the expected passage the top hit for a set of labelled queries (hit@1, hit@3);
* grounding with the honest backend: how many answers pass the checks without a fallback;
* guardrail catch rate: for each broken backend, how many of its answers are caught (not returned as model output).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iot_ai.ai.backends import MockBackend
from iot_ai.ai.embed import HashingEmbedder
from iot_ai.ai.engine import Engine
from iot_ai.ai.knowledge import build_index, load_passages
from iot_ai.ai.tools import DeviceStore
from iot_ai.config import PROCESSED_DIR, RESULTS_DIR

RETRIEVAL_CASES: list[tuple[str, str]] = [
    ("a sensor repeats the same temperature value for an hour", "sensor_faults#stuck-sensor"),
    ("one reading jumps far away from its neighbours and comes back", "sensor_faults#spike"),
    ("the device has no readings for a long stretch", "sensor_faults#dropout"),
    ("battery voltage is falling faster than its usual trend", "battery_and_power#voltage-trend"),
    ("when should I schedule a battery check", "battery_and_power#scheduling-a-battery-check"),
    ("the device is silent but the other devices nearby are fine", "connectivity#first-checks-for-a-silent-device"),
    ("timestamps jump backwards in time", "connectivity#clock-gaps"),
    ("temperature of 120 degrees is physically impossible", "sensor_faults#out-of-range-values"),
    ("what are the rules for a missing value", "data_quality_rules#rules-used-in-this-demo"),
    ("send a technician to look at the sensor element", "response_actions#inspect-sensor"),
    ("add a visit for the battery", "response_actions#schedule-battery-check"),
    ("no action now keep watching the device", "response_actions#monitor-only"),
]


def run_eval(processed_dir: Path = PROCESSED_DIR, results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    embedder = HashingEmbedder()
    passages = load_passages()
    index = build_index(passages, embedder)
    data = DeviceStore.load(processed_dir)

    hit1 = hit3 = 0
    retrieval_detail: list[dict[str, Any]] = []
    for query, expected in RETRIEVAL_CASES:
        ids = [h.id for h in index.search(embedder.embed([query])[0], 3)]
        hit1 += ids[0] == expected
        hit3 += expected in ids
        retrieval_detail.append({"query": query, "expected": expected, "top3": ids})

    events = json.loads((processed_dir / "events.json").read_text(encoding="utf-8"))
    answer_key = {
        row["event_id"]: row
        for row in json.loads((processed_dir / "answer_key.json").read_text(encoding="utf-8"))
    }
    sample = _sample_events(events, answer_key, 40)

    def run_with(mode: str) -> list[Any]:
        engine = Engine(MockBackend(mode), embedder, index, passages, data)
        return [engine.explain(e) for e in sample]

    good = run_with("good")
    good_ok = sum(1 for x in good if x.status == "model")
    bad_modes = ["wrong_number", "fake_citation", "no_tools", "unknown_tool"]
    catch: dict[str, dict[str, int]] = {}
    for mode in bad_modes:
        results = run_with(mode)
        caught = sum(1 for x in results if x.status == "fallback_template")
        catch[mode] = {"cases": len(results), "caught": caught}

    total_bad = sum(v["cases"] for v in catch.values())
    total_caught = sum(v["caught"] for v in catch.values())
    report: dict[str, Any] = {
        "events_evaluated": len(sample),
        "retrieval": {
            "queries": len(RETRIEVAL_CASES),
            "hit_at_1": round(hit1 / len(RETRIEVAL_CASES), 3),
            "hit_at_3": round(hit3 / len(RETRIEVAL_CASES), 3),
            "embedder": embedder.name,
            "detail": retrieval_detail,
        },
        "honest_backend": {
            "backend": "mock-good",
            "answers_passing_checks": good_ok,
            "answers_total": len(good),
            "numbers_checked_total": sum(x.numbers_checked for x in good),
        },
        "broken_backends": catch,
        "guardrail_catch_rate": round(total_caught / total_bad, 3) if total_bad else 0.0,
        "guardrail_caught": total_caught,
        "guardrail_cases": total_bad,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "engine_eval.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _sample_events(
    events: list[dict[str, Any]], answer_key: dict[str, dict[str, Any]], n: int
) -> list[dict[str, Any]]:
    """Deterministic sample: the first n events that have at least one rule fired, or an injected overlap.

    The answer key is used here, for evaluation only. The sampled events themselves carry no label fields.
    """
    picked = [
        e
        for e in events
        if e["rules_fired"] or answer_key.get(f"{e['device']}@{e['start']}", {}).get("injected_overlap", False)
    ]
    return picked[:n]
