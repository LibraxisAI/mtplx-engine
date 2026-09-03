"""Protocol-neutral generation turn event types.

The taxonomy covers the temporal lifecycle of a single generation turn:
TurnStarted, deltas (text, reasoning, tool-call), usage, and exactly one
terminal event (completed, failed, or cancelled).

Transition invariants:
  1. Exactly one TurnStarted, always first.
  2. OutputItemStarted before deltas of a new output kind (optional;
     encoders may infer boundaries from first-delta).
  3. No deltas after the terminal event.
  4. Exactly one terminal: TurnCompleted | TurnFailed | TurnCancelled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class TurnStarted:
    response_id: str
    model: str
    created: int
    timestamp_s: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True)
class OutputItemStarted:
    item_type: str
    output_index: int


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
class TurnCompleted:
    finish_reason: str
    usage: UsageUpdate | None = None


@dataclass(frozen=True)
class TurnFailed:
    error: str
    code: str | None = None


@dataclass(frozen=True)
class TurnCancelled:
    reason: str


TurnEvent = Union[
    TurnStarted,
    OutputItemStarted,
    TextDelta,
    ReasoningDelta,
    ToolCallDelta,
    UsageUpdate,
    TurnCompleted,
    TurnFailed,
    TurnCancelled,
]
