"""GenerationTurn — protocol-neutral lifecycle authority for a single generation.

State machine with enforced transitions:

    IDLE ──start()──▸ STARTED ──complete()/fail()/cancel()──▸ TERMINAL

  - ``start(TurnStarted)``  — IDLE → STARTED, exactly once.
  - ``emit(delta)``         — requires STARTED; silently dropped in TERMINAL.
  - ``complete/fail/cancel``— STARTED → TERMINAL, idempotent once terminal.
  - ``fail(TurnFailed)``    — also allowed from IDLE (immediate errors).

Terminal event closes the stream — there is no separate ``finish()``.  EOF
without a terminal event is structurally impossible.

Fan-out:  ``subscribe()`` returns a bounded ``AsyncIterator[TurnEvent]`` that
replays buffered events and receives live broadcasts.  Multiple subscribers
may observe the same turn concurrently.  A subscriber whose queue overflows
is dropped (generation never blocks on a slow consumer).

Thread-safety invariant:  all mutating calls (start/emit/complete/fail/cancel)
must execute on the owning asyncio event loop thread.  This holds in the
current codebase where every call site is inside Starlette's async-generator
iteration of ``event_stream()``.
"""

from __future__ import annotations

import asyncio
import enum
import time
from collections.abc import AsyncIterator

from .events import (
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    TurnCancelled,
    TurnCompleted,
    TurnEvent,
    TurnFailed,
    TurnStarted,
)

_TERMINAL_TYPES = (TurnCompleted, TurnFailed, TurnCancelled)
_DELTA_TYPES = (TextDelta, ReasoningDelta, ToolCallDelta)


class TurnState(enum.Enum):
    IDLE = "idle"
    STARTED = "started"
    TERMINAL = "terminal"


class GenerationTurn:

    _DEFAULT_MAXSIZE = 4096

    def __init__(self, *, maxsize: int = _DEFAULT_MAXSIZE) -> None:
        self._state = TurnState.IDLE
        self._events: list[TurnEvent] = []
        self._subscribers: list[asyncio.Queue[TurnEvent | None]] = []
        self._maxsize = maxsize
        self._first_delta_s: float | None = None

    @property
    def state(self) -> TurnState:
        return self._state

    @property
    def first_delta_s(self) -> float | None:
        return self._first_delta_s

    def start(self, event: TurnStarted) -> None:
        if self._state != TurnState.IDLE:
            raise RuntimeError(
                f"TurnStarted requires IDLE state, got {self._state.value}"
            )
        self._state = TurnState.STARTED
        self._broadcast(event)

    def emit(self, event: TurnEvent) -> None:
        if self._state == TurnState.TERMINAL:
            return
        if self._state != TurnState.STARTED:
            raise RuntimeError(
                f"emit requires STARTED state, got {self._state.value}"
            )
        if isinstance(event, (TurnStarted,) + _TERMINAL_TYPES):
            raise TypeError(
                f"use start()/complete()/fail()/cancel() for {type(event).__name__}"
            )
        if self._first_delta_s is None and isinstance(event, _DELTA_TYPES):
            self._first_delta_s = time.perf_counter()
        self._broadcast(event)

    def complete(self, event: TurnCompleted) -> None:
        self._terminal(event)

    def fail(self, event: TurnFailed) -> None:
        self._terminal(event)

    def cancel(self, event: TurnCancelled) -> None:
        self._terminal(event)

    def subscribe(self, *, maxsize: int | None = None) -> AsyncIterator[TurnEvent]:
        sz = maxsize if maxsize is not None else self._maxsize
        q: asyncio.Queue[TurnEvent | None] = asyncio.Queue(maxsize=sz)
        for ev in self._events:
            q.put_nowait(ev)
        if self._state == TurnState.TERMINAL:
            q.put_nowait(None)
        else:
            self._subscribers.append(q)
        return _SubscriberIterator(q)

    def _terminal(self, event: TurnEvent) -> None:
        if self._state == TurnState.TERMINAL:
            return
        if self._state == TurnState.IDLE and not isinstance(event, TurnFailed):
            raise RuntimeError(
                f"only TurnFailed is allowed from IDLE state, "
                f"got {type(event).__name__}"
            )
        self._state = TurnState.TERMINAL
        self._broadcast(event)
        self._close_subscribers()

    def _broadcast(self, event: TurnEvent) -> None:
        self._events.append(event)
        overflow: list[int] = []
        for i, q in enumerate(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                overflow.append(i)
        for i in reversed(overflow):
            q = self._subscribers.pop(i)
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass

    def _close_subscribers(self) -> None:
        for q in self._subscribers:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass
        self._subscribers.clear()


class _SubscriberIterator:

    __slots__ = ("_queue",)

    def __init__(self, queue: asyncio.Queue[TurnEvent | None]) -> None:
        self._queue = queue

    def __aiter__(self) -> AsyncIterator[TurnEvent]:
        return self

    async def __anext__(self) -> TurnEvent:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item
