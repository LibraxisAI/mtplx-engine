"""Protocol-neutral generation turn event types.

The taxonomy covers the temporal lifecycle of a single generation turn:
TurnStarted, deltas (text, reasoning, tool-call), usage, and exactly one
terminal event (completed, failed, or cancelled).

Transition invariants (enforced by GenerationTurn state machine):
  1. Exactly one TurnStarted, always first.
  2. Deltas only between start and terminal.
  3. Exactly one terminal: TurnCompleted | TurnFailed | TurnCancelled.
  4. Terminal closes the event stream — no separate finish required.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(frozen=True)
class TurnStarted:
    response_id: str
    model: str
    created: int
    timestamp_s: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True)
class OutputItemStarted:
    kind: str
    index: int = 0
    item_id: str | None = None


@dataclass(frozen=True)
class TextDelta:
    delta: str


@dataclass(frozen=True)
class ReasoningDelta:
    delta: str


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


@dataclass(frozen=True)
class UsageUpdate:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class TurnHeartbeat:
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class TurnCompleted:
    finish_reason: str
    usage: UsageUpdate | None = None
    mtplx_stats: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None


@dataclass(frozen=True)
class TurnFailed:
    error: str
    code: str | None = None
    status_code: int = 500


@dataclass(frozen=True)
class TurnCancelled:
    reason: str


TerminalEvent = Union[TurnCompleted, TurnFailed, TurnCancelled]

TurnEvent = Union[
    TurnStarted,
    OutputItemStarted,
    TextDelta,
    ReasoningDelta,
    ToolCallDelta,
    UsageUpdate,
    TurnHeartbeat,
    TurnCompleted,
    TurnFailed,
    TurnCancelled,
]
