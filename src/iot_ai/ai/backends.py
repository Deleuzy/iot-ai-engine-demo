"""Model backends. One interface (Backend.turn), five implementations:

* MockBackend: deterministic, offline. Used in tests and CI. Its modes produce deliberately wrong
  outputs, so the guardrail tests can show that the checks catch them.
* AnthropicBackend: Anthropic Messages API with tool use.
* OpenAICompatBackend: any OpenAI-style /chat/completions endpoint with tools (vLLM serving Qwen or Llama,
  Mistral's API, OpenAI itself).

Keys are read from the environment and never written to logs or results.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

import httpx

from iot_ai.ai.types import Message, ModelTurn, ToolCall, ToolSpec

WINDOW_START = datetime(2004, 2, 28)
WINDOW_END = datetime(2004, 4, 6, 23, 59)


class BackendError(RuntimeError):
    pass


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BackendError(f"tool arguments are not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BackendError("tool arguments must be a JSON object")
    return value


class MockBackend:
    """A fixed policy that reads the event from the user message and answers from the tool results.

    mode: good | wrong_number | fake_citation | no_tools | unknown_tool
    The bad modes exist only to test the guardrails. Each one must be caught before an answer is returned.
    """

    def __init__(self, mode: str = "good") -> None:
        self.mode = mode
        self.name = f"mock-{mode}"

    def turn(self, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        user_text = next(m.text for m in messages if m.role == "user")
        event = json.loads(user_text.split("EVENT_JSON:", 1)[1].split("\n", 1)[0])
        passage_ids = json.loads(user_text.split("PASSAGE_IDS:", 1)[1].split("\n", 1)[0])
        results = [m for m in messages if m.role == "tool"]
        device = int(event["device"])
        if self.mode == "unknown_tool" and not results:
            return ModelTurn("", [ToolCall("u1", "delete_device", {"device": device})])
        if not results and self.mode != "no_tools":
            start = datetime.fromisoformat(str(event["start"])) - timedelta(days=1)
            end = datetime.fromisoformat(str(event["end"])) + timedelta(days=1)
            start = max(start, WINDOW_START)
            end = min(end, WINDOW_END)
            return ModelTurn(
                "",
                [
                    ToolCall("c1", "get_device_history", {"device": device, "start": str(start), "end": str(end)}),
                    ToolCall("c2", "get_device_events", {"device": device}),
                ],
            )
        history: dict[str, Any] = {}
        for m in results:
            payload = json.loads(m.text)
            if "bins" in payload:
                history = payload
        action, cited = self._runbook(event, passage_ids)
        rules = ", ".join(event["rules_fired"]) if event["rules_fired"] else "none (model score only)"
        if history:
            temp = (
                f"a mean temperature of {history['temperature_mean']} C"
                if history["temperature_mean"] is not None
                else "no valid temperature reading"
            )
            battery = (
                f"its battery voltage moved from {history['voltage_first']} V to {history['voltage_last']} V, "
                f"a change of {history['voltage_change_per_day']} V per day"
                if history["voltage_first"] is not None
                else "no battery voltage reading"
            )
            context = f"In the surrounding window the device had {temp} and {battery}."
        else:
            # Tool failed (mock unknown_tool mode): answer from the event only; the engine must still refuse it.
            context = "No device history was available."
        summary = (
            f"Device {device} was flagged from {event['start']} to {event['end']} over {event['bins']} bins "
            f"with score {event['max_score']}. Rules that fired: {rules}. {context}"
        )
        if self.mode == "wrong_number":
            # A number that no tool returned, appended after the grounded text.
            summary += " The device ran 99.9 degrees hotter than its neighbours."
        if self.mode == "fake_citation":
            cited = [*cited, "invented_document#made-up-section"]
        payload_out = {"summary": summary, "suggested_action": action, "cited_passages": cited}
        return ModelTurn(json.dumps(payload_out))

    @staticmethod
    def _runbook(event: dict[str, Any], passage_ids: list[str]) -> tuple[str, list[str]]:
        rules = set(event["rules_fired"])
        if "missing" in rules:
            action, topic = "check_connectivity_and_power", "connectivity"
        elif "stuck" in rules:
            action, topic = "inspect_sensor", "sensor_faults"
        elif "spike" in rules:
            action, topic = "verify_reading_and_resample", "sensor_faults"
        elif "range" in rules:
            action, topic = "inspect_sensor", "data_quality_rules"
        else:
            action, topic = "monitor_only", "response_actions"
        action_id = f"response_actions#{action.replace('_', '-')}"
        cited = [pid for pid in passage_ids if pid.startswith(f"{topic}#")][:1]
        if action_id in passage_ids:
            cited.append(action_id)
        if not cited:
            cited = passage_ids[:1]  # cite what was retrieved, never an id invented here
        return action, cited


class OpenAICompatBackend:
    name = "openai-compatible"

    def __init__(
        self, base_url: str, model: str, api_key: str = "", transport: httpx.BaseTransport | None = None
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=120.0, transport=transport)

    def turn(self, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        wire: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "assistant" and m.tool_calls:
                wire.append(
                    {
                        "role": "assistant",
                        "content": m.text or None,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                            }
                            for c in m.tool_calls
                        ],
                    }
                )
            elif m.role == "tool":
                wire.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.text})
            else:
                wire.append({"role": m.role, "content": m.text})
        body = {
            "model": self.model,
            "messages": wire,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ],
            "temperature": 0,
            "max_tokens": 800,
        }
        response = self._client.post("/chat/completions", json=body)
        if response.status_code >= 400:
            raise BackendError(f"{self.name} returned HTTP {response.status_code}")
        message = response.json()["choices"][0]["message"]
        calls = [
            ToolCall(
                id=str(c["id"]),
                name=str(c["function"]["name"]),
                arguments=_parse_json_object(c["function"]["arguments"]),
            )
            for c in (message.get("tool_calls") or [])
        ]
        return ModelTurn(str(message.get("content") or ""), calls)


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str, api_key: str, transport: httpx.BaseTransport | None = None) -> None:
        self.model = model
        self._client = httpx.Client(
            base_url="https://api.anthropic.com",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=120.0,
            transport=transport,
        )

    def turn(self, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        system = "\n".join(m.text for m in messages if m.role == "system")
        wire: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                wire.append(
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.text}],
                    }
                )
            elif m.role == "assistant" and m.tool_calls:
                blocks: list[dict[str, Any]] = [{"type": "text", "text": m.text}] if m.text else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
                wire.append({"role": "assistant", "content": blocks})
            else:
                wire.append({"role": m.role, "content": m.text})
        body = {
            "model": self.model,
            "max_tokens": 800,
            "temperature": 0,
            "system": system,
            "messages": wire,
            "tools": [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools],
        }
        response = self._client.post("/v1/messages", json=body)
        if response.status_code >= 400:
            raise BackendError(f"{self.name} returned HTTP {response.status_code}")
        blocks_out = response.json().get("content", [])
        text = "".join(b.get("text", "") for b in blocks_out if b.get("type") == "text")
        calls = [
            ToolCall(id=str(b["id"]), name=str(b["name"]), arguments=dict(b.get("input", {})))
            for b in blocks_out
            if b.get("type") == "tool_use"
        ]
        return ModelTurn(text, calls)


def make_backend(env: dict[str, str] | None = None) -> Any:
    """Choose the backend from ENGINE_BACKEND: mock (default), anthropic, mistral, openai_compat."""
    env = dict(os.environ) if env is None else env
    kind = env.get("ENGINE_BACKEND", "mock")
    model = env.get("ENGINE_MODEL", "")
    if kind == "mock":
        return MockBackend("good")
    if kind == "anthropic":
        key = env.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise BackendError("ANTHROPIC_API_KEY is not set")
        return AnthropicBackend(model or "claude-sonnet-5-5", key)
    if kind == "mistral":
        key = env.get("MISTRAL_API_KEY", "")
        if not key:
            raise BackendError("MISTRAL_API_KEY is not set")
        return OpenAICompatBackend("https://api.mistral.ai/v1", model or "mistral-large-latest", key)
    if kind == "openai_compat":
        base = env.get("ENGINE_BASE_URL", "")
        if not base:
            raise BackendError("ENGINE_BASE_URL is required, e.g. http://vllm:8000/v1")
        return OpenAICompatBackend(base, model or "Qwen/Qwen2.5-7B-Instruct", env.get("ENGINE_API_KEY", ""))
    raise BackendError(f"unknown ENGINE_BACKEND {kind}")
