from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from iot_ai.ai import guardrails
from iot_ai.ai.backends import AnthropicBackend, BackendError, MockBackend, OpenAICompatBackend, make_backend
from iot_ai.ai.embed import HashingEmbedder
from iot_ai.ai.engine import Engine
from iot_ai.ai.knowledge import build_index, load_passages
from iot_ai.ai.tools import DeviceStore, ToolError, run_tool
from iot_ai.ai.types import Message, ModelTurn
from iot_ai.api import State, create_app
from iot_ai.config import KNOWLEDGE_DIR


@pytest.fixture()
def events(processed_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads((processed_dir / "events.json").read_text(encoding="utf-8"))
    return rows


def make_engine(mode: str, store: DeviceStore) -> Engine:
    embedder = HashingEmbedder()
    passages = load_passages(KNOWLEDGE_DIR)
    return Engine(MockBackend(mode), embedder, build_index(passages, embedder), passages, store)


def test_knowledge_loads_sections() -> None:
    passages = load_passages(KNOWLEDGE_DIR)
    ids = {p.id for p in passages}
    assert "sensor_faults#stuck-sensor" in ids
    assert "response_actions#monitor-only" in ids
    assert len(passages) >= 15


def test_retrieval_finds_the_stuck_passage() -> None:
    embedder = HashingEmbedder()
    passages = load_passages(KNOWLEDGE_DIR)
    index = build_index(passages, embedder)
    top = index.search(embedder.embed(["a sensor repeats the same temperature value for an hour"])[0], 3)
    assert "sensor_faults#stuck-sensor" in [h.id for h in top]


def test_good_mock_answer_passes_and_uses_tools(store: DeviceStore, events: list[dict[str, Any]]) -> None:
    out = make_engine("good", store).explain(events[0])
    assert out.status == "model"
    assert out.tool_calls == 2
    assert out.suggested_action == "inspect_sensor"
    assert out.numbers_checked > 0
    assert all(c.startswith(("sensor_faults#", "response_actions#")) for c in out.cited_passages)


@pytest.mark.parametrize("mode", ["wrong_number", "fake_citation", "no_tools", "unknown_tool"])
def test_broken_backends_fall_back(store: DeviceStore, events: list[dict[str, Any]], mode: str) -> None:
    out = make_engine(mode, store).explain(events[0])
    assert out.status == "fallback_template"
    assert out.guardrail_findings


def test_fallback_has_no_unchecked_numbers(store: DeviceStore, events: list[dict[str, Any]]) -> None:
    out = make_engine("wrong_number", store).explain(events[1])
    assert out.status == "fallback_template"
    assert out.numbers_checked == 0
    assert out.suggested_action == "check_connectivity_and_power"


def test_tool_argument_validation(store: DeviceStore) -> None:
    with pytest.raises(ToolError):
        run_tool(store, "get_device_history", {"device": "1", "start": "x", "end": "y"})
    with pytest.raises(ToolError):
        run_tool(store, "get_device_history", {"device": 999, "start": "2004-03-01", "end": "2004-03-02"})
    with pytest.raises(ToolError):
        run_tool(store, "delete_device", {"device": 1})


def test_history_numbers(store: DeviceStore) -> None:
    h = run_tool(
        store, "get_device_history", {"device": 2, "start": "2004-03-02 00:00:00", "end": "2004-03-02 23:50:00"}
    )
    assert h["bins_missing"] == 12
    assert h["longest_silence_bins"] == 12


def test_backend_factory_requires_keys() -> None:
    with pytest.raises(BackendError):
        make_backend({"ENGINE_BACKEND": "anthropic"})
    assert isinstance(make_backend({}), MockBackend)


def test_openai_compat_backend_parses_tool_calls() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "t1",
                                    "type": "function",
                                    "function": {"name": "get_device_events", "arguments": '{"device": 3}'},
                                }
                            ],
                        }
                    }
                ]
            },
        )

    backend = OpenAICompatBackend("http://vllm:8000/v1", "qwen", transport=httpx.MockTransport(handler))
    turn = backend.turn([Message(role="system", text="s"), Message(role="user", text="u")], [])
    assert turn.tool_calls[0].name == "get_device_events"
    assert turn.tool_calls[0].arguments == {"device": 3}
    assert seen["body"]["model"] == "qwen"


def test_anthropic_backend_parses_tool_use() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "test-key"
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "checking"},
                    {
                        "type": "tool_use",
                        "id": "tu1",
                        "name": "get_device_history",
                        "input": {"device": 1, "start": "a", "end": "b"},
                    },
                ]
            },
        )

    backend = AnthropicBackend("claude-test", "test-key", transport=httpx.MockTransport(handler))
    turn = backend.turn([Message(role="system", text="s"), Message(role="user", text="u")], [])
    assert turn.text == "checking"
    assert turn.tool_calls[0].arguments["device"] == 1


def test_api_explain_and_health(store: DeviceStore, processed_dir: Path, events: list[dict[str, Any]]) -> None:
    embedder = HashingEmbedder()
    passages = load_passages(KNOWLEDGE_DIR)
    engine = Engine(MockBackend("good"), embedder, build_index(passages, embedder), passages, store)
    client = TestClient(create_app(State(engine, events, "mock-good")))
    assert client.get("/health").json()["data_loaded"] is True
    listed = client.get("/events", params={"device": 1}).json()
    assert listed and listed[0]["event_id"] == "1@2004-03-01 08:20:00"
    body = client.post("/explain", json={"event_id": listed[0]["event_id"]}).json()
    assert body["status"] == "model"
    assert client.post("/explain", json={"event_id": "nope"}).status_code == 404


def test_api_without_data_returns_503() -> None:
    client = TestClient(create_app(State(None, [], "none")))
    assert client.post("/explain", json={"event_id": "x"}).status_code == 503


class _Recorder:
    name = "recorder"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def turn(self, messages: list[Message], tools: list[Any]) -> ModelTurn:
        self.prompts.append(next(m.text for m in messages if m.role == "user"))
        return ModelTurn("not an answer")


class _Raises:
    name = "raises"

    def turn(self, messages: list[Message], tools: list[Any]) -> ModelTurn:
        raise httpx.ConnectError("provider unreachable")


def test_prompt_has_no_answer_key(store: DeviceStore, events: list[dict[str, Any]]) -> None:
    embedder = HashingEmbedder()
    passages = load_passages(KNOWLEDGE_DIR)
    recorder = _Recorder()
    engine = Engine(recorder, embedder, build_index(passages, embedder), passages, store)
    leaky = next(e for e in events if e["injected_overlap"])
    engine.explain(leaky)
    assert recorder.prompts
    assert "injected" not in recorder.prompts[0]
    assert "fault_kind" not in recorder.prompts[0]


def test_backend_exception_falls_back(store: DeviceStore, events: list[dict[str, Any]]) -> None:
    embedder = HashingEmbedder()
    passages = load_passages(KNOWLEDGE_DIR)
    engine = Engine(_Raises(), embedder, build_index(passages, embedder), passages, store)
    out = engine.explain(events[0])
    assert out.status == "fallback_template"
    assert any("backend failed" in f for f in out.guardrail_findings)


@pytest.mark.parametrize("rules", [["missing"], ["stuck"], ["spike"], ["range"], []])
def test_fallback_meets_the_same_checks(store: DeviceStore, events: list[dict[str, Any]], rules: list[str]) -> None:
    """The template must pass the guardrail checks by construction: its citation is retrieved, its numbers come from the event."""
    embedder = HashingEmbedder()
    passages = load_passages(KNOWLEDGE_DIR)
    engine = Engine(MockBackend("good"), embedder, build_index(passages, embedder), passages, store)
    event = {**events[0], "rules_fired": rules}
    retrieved = engine.retrieve(event)
    out = engine._fallback("x", event, retrieved, [], 1)
    answer = {"summary": out.summary, "suggested_action": out.suggested_action, "cited_passages": out.cited_passages}
    verdict = guardrails.check(answer, [], retrieved, event, 1, [])
    assert verdict.passed, verdict.findings


def test_bad_timestamp_is_a_tool_error(store: DeviceStore) -> None:
    with pytest.raises(ToolError):
        run_tool(store, "get_device_history", {"device": 1, "start": "yesterday", "end": "today"})


def test_device_events_are_the_most_recent(store: DeviceStore) -> None:
    rows = run_tool(store, "get_device_events", {"device": 1})["events"]
    starts = [r["start"] for r in rows]
    assert starts == sorted(starts)
