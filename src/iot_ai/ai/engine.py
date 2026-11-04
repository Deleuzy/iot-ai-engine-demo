"""The engine: retrieve passages, let the model call tools, check the answer, or fall back to a template.

Model-independent: it only sees Backend.turn(). A backend that breaks a rule cannot return an unchecked answer;
the engine returns the template explanation instead, and records why.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from iot_ai.ai import guardrails
from iot_ai.ai.embed import Embedder
from iot_ai.ai.knowledge import Passage
from iot_ai.ai.store import VectorStore
from iot_ai.ai.tools import TOOL_NAMES, TOOL_SPECS, DeviceStore, ToolError, run_tool
from iot_ai.ai.types import Backend, Message, ToolCall

MAX_ROUNDS = 4
# The only event fields the model may see. Anything else (for example answer-key fields) is left out.
PUBLIC_EVENT_FIELDS = ("device", "start", "end", "bins", "max_score", "rules_fired")
SYSTEM_PROMPT = (
    "You explain one flagged event from an IoT sensor device. Use the tools to read the device data before you "
    "answer. Only state numbers that a tool or the event returned. Cite only the passage ids you were given. "
    "Answer with a JSON object with keys summary (text), suggested_action (one of the listed actions) and "
    "cited_passages (list of passage ids)."
)


@dataclass
class Explanation:
    event_id: str
    status: str  # model | fallback_template
    summary: str
    suggested_action: str
    cited_passages: list[str]
    backend: str
    tool_calls: int
    retrieved_passages: list[str]
    guardrail_findings: list[str] = field(default_factory=list)
    numbers_checked: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Engine:
    def __init__(
        self, backend: Backend, embedder: Embedder, index: VectorStore, passages: list[Passage], data: DeviceStore
    ) -> None:
        self.backend = backend
        self.embedder = embedder
        self.index = index
        self.passages = {p.id: p for p in passages}
        self.data = data

    def retrieve(self, event: dict[str, Any], k: int = 4) -> list[str]:
        rules = " ".join(event["rules_fired"]) or "model score only"
        query = f"sensor device fault {rules} battery voltage connectivity runbook action"
        vec = self.embedder.embed([query])[0]
        return [h.id for h in self.index.search(vec, k)]

    def _run_tools(self, calls: list[ToolCall], outputs: list[Any], trace: list[Message], errors: list[str]) -> None:
        for call in calls:
            try:
                if call.name not in TOOL_NAMES:
                    raise ToolError(f"tool {call.name} is not available")
                result = run_tool(self.data, call.name, call.arguments)
                outputs.append(result)
                text = json.dumps(result, default=str)
            except ToolError as exc:
                errors.append(str(exc))
                text = json.dumps({"error": str(exc)})
            trace.append(Message(role="tool", text=text, tool_call_id=call.id, tool_name=call.name))

    def explain(self, event: dict[str, Any]) -> Explanation:
        event_id = f"{event['device']}@{event['start']}"
        retrieved = self.retrieve(event)
        passage_block = "\n".join(
            f"- {pid}: {self.passages[pid].title}. {self.passages[pid].text}" for pid in retrieved
        )
        public_event = {k: event[k] for k in PUBLIC_EVENT_FIELDS}
        user = (
            f"EVENT_JSON: {json.dumps(public_event)}\n"
            f"PASSAGE_IDS: {json.dumps(retrieved)}\n"
            f"ACTIONS: {json.dumps(list(guardrails.ACTIONS))}\n"
            f"PASSAGES:\n{passage_block}"
        )
        trace: list[Message] = [Message(role="system", text=SYSTEM_PROMPT), Message(role="user", text=user)]
        outputs: list[Any] = []
        errors: list[str] = []
        tool_calls = 0
        answer_text = ""
        for _ in range(MAX_ROUNDS):
            try:
                turn = self.backend.turn(trace, TOOL_SPECS)
            except Exception as exc:  # a provider or transport failure is not an answer: fall back, never crash
                errors.append(f"backend failed: {type(exc).__name__}")
                break
            if not turn.tool_calls:
                answer_text = turn.text
                break
            tool_calls += len(turn.tool_calls)
            trace.append(Message(role="assistant", text=turn.text, tool_calls=tuple(turn.tool_calls)))
            self._run_tools(turn.tool_calls, outputs, trace, errors)
        parsed = guardrails.parse_answer(answer_text)
        verdict = guardrails.check(parsed, outputs, retrieved, event, tool_calls, errors)
        if verdict.passed and parsed is not None:
            return Explanation(
                event_id=event_id,
                status="model",
                summary=str(parsed["summary"]),
                suggested_action=str(parsed["suggested_action"]),
                cited_passages=[str(c) for c in parsed["cited_passages"]],
                backend=self.backend.name,
                tool_calls=tool_calls,
                retrieved_passages=retrieved,
                guardrail_findings=[],
                numbers_checked=verdict.numbers_checked,
            )
        return self._fallback(event_id, event, retrieved, verdict.findings, tool_calls)

    def _fallback(
        self, event_id: str, event: dict[str, Any], retrieved: list[str], findings: list[str], tool_calls: int
    ) -> Explanation:
        """Template text built only from the event fields. It always passes the same checks by construction."""
        rules = event["rules_fired"]
        action = (
            "monitor_only"
            if not rules
            else (
                "check_connectivity_and_power"
                if "missing" in rules
                else "inspect_sensor"
                if "stuck" in rules or "range" in rules
                else "verify_reading_and_resample"
            )
        )
        # Cite only a passage that was retrieved for this request, so the fallback meets the same citation check.
        preferred = f"response_actions#{action.replace('_', '-')}"
        cite_id = preferred if preferred in retrieved else retrieved[0]
        summary = (
            f"Device {event['device']} was flagged from {event['start']} to {event['end']} over {event['bins']} bins "
            f"with score {event['max_score']}. Rules that fired: {', '.join(rules) if rules else 'no rules'}. "
            "The assistant could not verify its own answer, so this is the standard template explanation."
        )
        return Explanation(
            event_id=event_id,
            status="fallback_template",
            summary=summary,
            suggested_action=action,
            cited_passages=[cite_id],
            backend=self.backend.name,
            tool_calls=tool_calls,
            retrieved_passages=retrieved,
            guardrail_findings=findings,
            numbers_checked=0,
        )
