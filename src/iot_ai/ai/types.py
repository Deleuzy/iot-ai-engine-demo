"""Model-independent types. Every backend returns a ModelTurn; nothing else in the engine knows a provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """One step of the conversation in the engine's own format. Backends convert it to their wire format."""

    role: str  # system | user | assistant | tool
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""
    tool_name: str = ""


@dataclass(frozen=True)
class ModelTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class Backend(Protocol):
    name: str

    def turn(self, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        """One model call. The system prompt is the first message with role 'system'."""
        ...
