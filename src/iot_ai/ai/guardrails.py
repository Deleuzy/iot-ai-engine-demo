"""Output checks written in code. The model's answer is accepted only if every check passes.

1. The answer is JSON with the three fields, a bounded summary, and a suggested action from the fixed list.
2. The model called at least one tool before answering (no answer from memory alone).
3. Every cited passage id was actually retrieved for this request.
4. Every number in the summary matches a number a tool or the event returned, at the precision written.
   ISO dates and times are excluded, since they are identifiers. Nothing else is excluded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

ACTIONS = (
    "inspect_sensor",
    "check_connectivity_and_power",
    "verify_reading_and_resample",
    "schedule_battery_check",
    "monitor_only",
)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?")


@dataclass
class Verdict:
    passed: bool
    findings: list[str] = field(default_factory=list)
    numbers_checked: int = 0


def parse_answer(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _numbers_in_tool_output(value: Any, out: list[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, dict):
        for v in value.values():
            _numbers_in_tool_output(v, out)
    elif isinstance(value, list):
        for v in value:
            _numbers_in_tool_output(v, out)


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def _matches(token: str, allowed: list[float]) -> bool:
    value = float(token)
    d = _decimals(token)
    tolerance = 0.5 * 10 ** (-d) + 1e-9
    return any(abs(a - value) <= tolerance for a in allowed)


def findings_blocking(findings: list[str]) -> bool:
    """Number checks are skipped when the answer is already rejected for structure; this keeps reports short."""
    return any(f.startswith("tool call failed") for f in findings)


def check(
    answer: dict[str, Any] | None,
    tool_outputs: list[Any],
    retrieved_ids: list[str],
    event: dict[str, Any],
    tool_calls_made: int,
    tool_errors: list[str],
) -> Verdict:
    findings: list[str] = []
    findings.extend(f"tool call failed: {e}" for e in tool_errors)
    if answer is None:
        return Verdict(False, ["answer is not a JSON object"])
    summary = answer.get("summary")
    action = answer.get("suggested_action")
    cited = answer.get("cited_passages")
    if not isinstance(summary, str) or not (20 <= len(summary) <= 800):
        findings.append("summary must be text of 20 to 800 characters")
    if action not in ACTIONS:
        findings.append(f"suggested_action must be one of {list(ACTIONS)}")
    if not isinstance(cited, list) or not cited or not all(isinstance(c, str) for c in cited):
        findings.append("cited_passages must be a non-empty list of ids")
    else:
        unknown = [c for c in cited if c not in retrieved_ids]
        if unknown:
            findings.append(f"cited passages were not retrieved: {unknown}")
    if tool_calls_made == 0 and not tool_errors:
        findings.append("no tool was called, so the answer is not grounded in device data")
    numbers_checked = 0
    if isinstance(summary, str) and not findings_blocking(findings):
        allowed: list[float] = []
        for out in tool_outputs:
            _numbers_in_tool_output(out, allowed)
        _numbers_in_tool_output({k: v for k, v in event.items() if isinstance(v, (int, float))}, allowed)
        prose = _ISO.sub(" ", summary)
        if re.search(r"\b(nan|none|inf|infinity)\b", prose, flags=re.IGNORECASE):
            findings.append("summary contains an empty or invalid value (nan, none or inf)")
        for token in _NUMBER.findall(prose):
            numbers_checked += 1
            if not _matches(token, allowed):
                findings.append(f"number {token} in the summary does not come from a tool result")
    return Verdict(not findings, findings, numbers_checked)
