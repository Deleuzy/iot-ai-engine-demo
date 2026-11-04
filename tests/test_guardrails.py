from __future__ import annotations

from typing import Any

from iot_ai.ai import guardrails

EVENT = {
    "device": 7,
    "start": "2004-03-10 01:00:00",
    "end": "2004-03-10 03:00:00",
    "bins": 12,
    "max_score": 0.8731,
    "rules_fired": ["stuck"],
}
TOOL = [{"device": 7, "bins": 300, "temperature_mean": 21.37, "voltage_last": 2.712}]
GOOD = {
    "summary": "Device 7 was flagged with score 0.8731 over 12 bins. Mean temperature was 21.37 C and voltage 2.712 V.",
    "suggested_action": "inspect_sensor",
    "cited_passages": ["sensor_faults#stuck-sensor"],
}


def run(answer: dict[str, Any], calls: int = 1, errors: list[str] | None = None) -> guardrails.Verdict:
    return guardrails.check(answer, TOOL, ["sensor_faults#stuck-sensor"], EVENT, calls, errors or [])


def test_grounded_answer_passes() -> None:
    verdict = run(GOOD)
    assert verdict.passed, verdict.findings
    assert verdict.numbers_checked >= 3


def test_number_not_from_tools_is_caught() -> None:
    bad = {**GOOD, "summary": "Device 7 was flagged with score 0.8731. Mean temperature was 22.37 C, a long period."}
    verdict = run(bad)
    assert not verdict.passed
    assert any("22.37" in f for f in verdict.findings)


def test_dates_are_not_numbers_to_check() -> None:
    ok = {**GOOD, "summary": "Flagged on 2004-03-10 01:00:00 with score 0.8731 and 21.37 C mean temperature."}
    assert run(ok).passed


def test_wrong_precision_is_caught() -> None:
    bad = {**GOOD, "summary": "Mean temperature was 21.6 C during a flagged stuck period of two hours."}
    assert not run(bad).passed


def test_unretrieved_citation_is_caught() -> None:
    bad = {**GOOD, "cited_passages": ["made_up#section"]}
    assert not run(bad).passed


def test_no_tool_call_is_caught() -> None:
    assert not run(GOOD, calls=0).passed


def test_unknown_action_is_caught() -> None:
    assert not run({**GOOD, "suggested_action": "reboot_cluster"}).passed


def test_tool_error_blocks_answer() -> None:
    assert not run(GOOD, errors=["tool delete_device is not available"]).passed


def test_parse_answer_handles_code_fence() -> None:
    parsed = guardrails.parse_answer('```json\n{"summary": "x"}\n```')
    assert parsed == {"summary": "x"}
    assert guardrails.parse_answer("not json") is None


def test_nan_in_summary_is_caught() -> None:
    bad = {**GOOD, "summary": "Mean temperature was nan C over the window, voltage 2.712 V with score 0.8731."}
    verdict = run(bad)
    assert not verdict.passed
    assert any("nan" in f for f in verdict.findings)


def test_glued_unit_numbers_are_checked() -> None:
    assert not run({**GOOD, "summary": "Device 7 logged 999bins of flagged time with score 0.8731 and 21.37 C."}).passed
    assert not run({**GOOD, "summary": "Voltage fell 3.7V with score 0.8731 and a mean of 21.37 C in the window."}).passed


def test_integer_must_match_a_tool_value_exactly() -> None:
    # 3 is within half a unit of 2.712 and 21 of 21.37, but neither is what the tools returned.
    assert not run({**GOOD, "summary": "Device 7 ran 3 C hotter than neighbours with score 0.8731 and 21.37 C."}).passed
    assert not run({**GOOD, "summary": "Device 7 mean temperature 21 C with score 0.8731 and voltage 2.712 V."}).passed


def test_decimal_rounded_to_its_own_precision_passes() -> None:
    assert run({**GOOD, "summary": "Device 7 had score 0.87 and mean temperature 21.4 C over the window."}).passed
