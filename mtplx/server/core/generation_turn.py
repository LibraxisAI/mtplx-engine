"""GenerationTurn — protocol-neutral lifecycle authority for a single generation.

State machine with enforced transitions:

    IDLE ──start()──▸ STARTED ──complete()/fail()/cancel()──▸ TERMINAL

  - ``start(TurnStarted)``  — IDLE → STARTED, exactly once.
  - ``emit(delta)``         — requires STARTED.
  - ``complete/fail/cancel``— STARTED → TERMINAL, exactly once.
  - ``fail(TurnFailed)``    — also allowed from IDLE (immediate errors).

Terminal event closes the stream — there is no separate ``finish()``.  EOF
without a terminal event is structurally impossible for subscribers.

Fan-out:  ``subscribe()`` returns a bounded ``AsyncIterator[TurnEvent]``.
Subscribers should attach before generation starts. Late subscribers receive
only the bounded replay window: the start event, recent events that fit, and
the terminal event when present. A live subscriber overflow is an explicit
TurnFailed terminal, never a silent clean EOF.

Thread-safety invariant:  all mutating calls (start/emit/complete/fail/cancel)
must execute on the owning asyncio event loop thread.  This holds in the
current codebase where every call site is inside Starlette's async-generator
iteration of ``event_stream()``.
"""

from __future__ import annotations

import asyncio
import enum
import time
from collections import deque
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

    def __init__(self, *, maxsize: int = _DEFAULT_MAXSIZE, replay: int | None = None) -> None:
        self._state = TurnState.IDLE
        self._start_event: TurnStarted | None = None
        self._terminal_event: TurnEvent | None = None
        self._history: deque[TurnEvent] = deque(maxlen=max(1, replay or maxsize))
        self._subscribers: list[asyncio.Queue[TurnEvent | None]] = []
        self._maxsize = max(2, int(maxsize))
        self._first_delta_s: float | None = None
        self._driver_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> TurnState:
        return self._state

    @property
    def first_delta_s(self) -> float | None:
        return self._first_delta_s

    @property
    def driver_task(self) -> asyncio.Task[None] | None:
        return self._driver_task

    def attach_driver(self, task: asyncio.Task[None]) -> None:
        if self._driver_task is not None:
            raise RuntimeError("generation driver already attached")
        self._driver_task = task

    def cancel_driver(self) -> None:
        task = self._driver_task
        if task is not None and not task.done():
            task.cancel()

    def start(self, event: TurnStarted) -> None:
        if self._state != TurnState.IDLE:
            raise RuntimeError(
                f"TurnStarted requires IDLE state, got {self._state.value}"
            )
        self._state = TurnState.STARTED
        self._start_event = event
        self._broadcast(event)

    def emit(self, event: TurnEvent) -> None:
        if self._state == TurnState.TERMINAL:
            raise RuntimeError("cannot emit after terminal event")
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
        sz = max(2, int(maxsize if maxsize is not None else self._maxsize))
        q: asyncio.Queue[TurnEvent | None] = asyncio.Queue(maxsize=sz)
        for ev in self._bounded_replay(sz):
            q.put_nowait(ev)
        if self._state == TurnState.TERMINAL:
            q.put_nowait(None)
        else:
            self._subscribers.append(q)
        return _SubscriberIterator(q)

    def _terminal(self, event: TurnEvent) -> None:
        if self._state == TurnState.TERMINAL:
            raise RuntimeError("terminal event already emitted")
        if self._state == TurnState.IDLE and not isinstance(event, TurnFailed):
            raise RuntimeError(
                f"only TurnFailed is allowed from IDLE state, "
                f"got {type(event).__name__}"
            )
        self._state = TurnState.TERMINAL
        self._terminal_event = event
        self._history.append(event)
        for q in self._subscribers:
            self._force_put(q, event)
            self._force_put(q, None)
        self._subscribers.clear()

    def _broadcast(self, event: TurnEvent) -> None:
        self._history.append(event)
        overflow: list[int] = []
        for i, q in enumerate(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                overflow.append(i)
        if overflow and self._state != TurnState.TERMINAL:
            self.fail(
                TurnFailed(
                    error="turn event subscriber overflow",
                    code="backpressure_overflow",
                )
            )

    def _close_subscribers(self) -> None:
        for q in self._subscribers:
            self._force_put(q, None)
        self._subscribers.clear()

    def _bounded_replay(self, maxsize: int) -> list[TurnEvent]:
        capacity = max(1, maxsize - 1)
        events = list(self._history)
        middle = [
            event
            for event in events
            if event is not self._start_event and event is not self._terminal_event
        ]
        replay: list[TurnEvent] = []
        if self._start_event is not None and capacity > 1:
            replay.append(self._start_event)
        terminal_slots = 1 if self._terminal_event is not None else 0
        middle_capacity = max(0, capacity - len(replay) - terminal_slots)
        replay.extend(middle[-middle_capacity:] if middle_capacity else [])
        if self._terminal_event is not None:
            replay.append(self._terminal_event)
        return replay[-capacity:]

    @staticmethod
    def _force_put(
        q: asyncio.Queue[TurnEvent | None], item: TurnEvent | None
    ) -> None:
        while True:
            try:
                q.put_nowait(item)
                return
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    continue


class _SubscriberIterator:

    __slots__ = ("_queue", "_closed")

    def __init__(self, queue: asyncio.Queue[TurnEvent | None]) -> None:
        self._queue = queue
        self._closed = False

    def __aiter__(self) -> AsyncIterator[TurnEvent]:
        return self

    async def __anext__(self) -> TurnEvent:
        if self._closed:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is None:
            self._closed = True
            raise StopAsyncIteration
        return item

    def drain_nowait(self) -> list[TurnEvent]:
        events: list[TurnEvent] = []
        if self._closed:
            return events
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return events
            if item is None:
                self._closed = True
                return events
            events.append(item)
