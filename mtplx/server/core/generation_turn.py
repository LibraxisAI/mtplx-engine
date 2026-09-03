"""GenerationTurn -- async iterator over protocol-neutral generation events.

The producer (consumer loop in chat_completions) calls ``emit()``; the
consumer (a wire-protocol encoder or a test harness) iterates via
``async for event in turn:``.  Generation stays off the event loop;
``emit`` is safe to call from any thread via ``put_nowait``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from .events import ReasoningDelta, TextDelta, ToolCallDelta, TurnEvent


class GenerationTurn:

    def __init__(self) -> None:
        self._queue: asyncio.Queue[TurnEvent | None] = asyncio.Queue()
        self._finished = False
        self._first_delta_s: float | None = None

    def emit(self, event: TurnEvent) -> None:
        if self._finished:
            return
        if self._first_delta_s is None and isinstance(
            event, (TextDelta, ReasoningDelta, ToolCallDelta)
        ):
            self._first_delta_s = time.perf_counter()
        self._queue.put_nowait(event)

    def finish(self) -> None:
        if not self._finished:
            self._finished = True
            self._queue.put_nowait(None)

    @property
    def first_delta_s(self) -> float | None:
        return self._first_delta_s

    def __aiter__(self) -> AsyncIterator[TurnEvent]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[TurnEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item
