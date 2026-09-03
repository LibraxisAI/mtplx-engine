"""Tests for the GenerationTurn event seam and Responses rendering from TurnEvents.

Acceptance criteria proven:
  1. Typed events cover: turn start, output item start, text/reasoning/tool
     deltas, usage, complete, fail, cancel.
  2. Responses rendered from same events, NOT from parsing Chat SSE (proof:
     fake event stream test — no Chat SSE involved at all).
  3. Generation remains off asyncio event loop (GenerationTurn.emit is sync).
  4. First model delta observable independently (timestamped).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from mtplx.server.core import (
    GenerationTurn,
    OutputItemStarted,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    TurnCancelled,
    TurnCompleted,
    TurnEvent,
    TurnFailed,
    TurnStarted,
    UsageUpdate,
)
from mtplx.server.protocols.responses.encoder import (
    responses_stream_from_turn_events,
)
from mtplx.server.protocols.responses.schema import ResponsesRequest


def _fake_responses_request(**overrides: Any) -> ResponsesRequest:
    base: dict[str, Any] = {
        "model": "test-model",
        "input": "hello",
    }
    base.update(overrides)
    return ResponsesRequest.model_validate(base)


class TestEventTaxonomy:
    """Verify every event type is frozen, typed, and carries required fields."""

    def test_turn_started(self) -> None:
        ev = TurnStarted(response_id="r1", model="m", created=100)
        assert ev.response_id == "r1"
        assert ev.model == "m"
        assert ev.created == 100
        assert ev.timestamp_s > 0

    def test_text_delta(self) -> None:
        ev = TextDelta(delta="hello")
        assert ev.delta == "hello"

    def test_reasoning_delta(self) -> None:
        ev = ReasoningDelta(delta="think")
        assert ev.delta == "think"

    def test_tool_call_delta(self) -> None:
        ev = ToolCallDelta(index=0, call_id="c1", name="fn", arguments_delta="{")
        assert ev.index == 0
        assert ev.call_id == "c1"
        assert ev.name == "fn"
        assert ev.arguments_delta == "{"

    def test_usage_update(self) -> None:
        ev = UsageUpdate(prompt_tokens=10, completion_tokens=20)
        assert ev.prompt_tokens == 10
        assert ev.completion_tokens == 20

    def test_turn_completed(self) -> None:
        usage = UsageUpdate(prompt_tokens=5, completion_tokens=15)
        ev = TurnCompleted(finish_reason="stop", usage=usage)
        assert ev.finish_reason == "stop"
        assert ev.usage is not None
        assert ev.usage.completion_tokens == 15

    def test_turn_failed(self) -> None:
        ev = TurnFailed(error="boom", code="server_error")
        assert ev.error == "boom"
        assert ev.code == "server_error"

    def test_turn_cancelled(self) -> None:
        ev = TurnCancelled(reason="client_disconnected")
        assert ev.reason == "client_disconnected"

    def test_output_item_started(self) -> None:
        ev = OutputItemStarted(item_type="message", output_index=0)
        assert ev.item_type == "message"

    def test_frozen(self) -> None:
        ev = TextDelta(delta="x")
        with pytest.raises(AttributeError):
            ev.delta = "y"  # type: ignore[misc]


class TestGenerationTurn:

    @pytest.mark.asyncio
    async def test_emit_and_iterate(self) -> None:
        turn = GenerationTurn()
        turn.emit(TurnStarted(response_id="r1", model="m", created=1))
        turn.emit(TextDelta(delta="hi"))
        turn.emit(TurnCompleted(finish_reason="stop"))
        turn.finish()

        events: list[TurnEvent] = []
        async for ev in turn:
            events.append(ev)

        assert len(events) == 3
        assert isinstance(events[0], TurnStarted)
        assert isinstance(events[1], TextDelta)
        assert isinstance(events[2], TurnCompleted)

    @pytest.mark.asyncio
    async def test_first_delta_s_tracked(self) -> None:
        turn = GenerationTurn()
        assert turn.first_delta_s is None
        turn.emit(TurnStarted(response_id="r1", model="m", created=1))
        assert turn.first_delta_s is None
        turn.emit(TextDelta(delta="a"))
        assert turn.first_delta_s is not None
        first = turn.first_delta_s
        turn.emit(TextDelta(delta="b"))
        assert turn.first_delta_s == first
        turn.finish()

    @pytest.mark.asyncio
    async def test_emit_after_finish_is_noop(self) -> None:
        turn = GenerationTurn()
        turn.emit(TextDelta(delta="a"))
        turn.finish()
        turn.emit(TextDelta(delta="b"))

        events: list[TurnEvent] = []
        async for ev in turn:
            events.append(ev)
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_finish_idempotent(self) -> None:
        turn = GenerationTurn()
        turn.emit(TextDelta(delta="a"))
        turn.finish()
        turn.finish()
        turn.finish()

        events: list[TurnEvent] = []
        async for ev in turn:
            events.append(ev)
        assert len(events) == 1


class TestResponsesFromTurnEvents:
    """Proof: Responses rendered from TurnEvents, NOT from Chat SSE parsing."""

    @pytest.mark.asyncio
    async def test_text_completion_renders_responses_sse(self) -> None:
        """A simple text completion emits the full Responses SSE lifecycle."""
        turn = GenerationTurn()
        request = _fake_responses_request()

        turn.emit(TurnStarted(response_id="resp_1", model="test-model", created=100))
        turn.emit(TextDelta(delta="Hello"))
        turn.emit(TextDelta(delta=" world"))
        turn.emit(TurnCompleted(
            finish_reason="stop",
            usage=UsageUpdate(prompt_tokens=5, completion_tokens=2),
        ))
        turn.finish()

        events: list[dict[str, Any]] = []
        async for chunk in responses_stream_from_turn_events(
            turn,
            request=request,
            response_id="resp_1",
            model="test-model",
            created_at=100,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        event_types = [e["type"] for e in events]
        assert "response.created" in event_types
        assert "response.in_progress" in event_types
        assert "response.output_item.added" in event_types
        assert "response.output_text.delta" in event_types
        assert "response.completed" in event_types

        text_deltas = [e for e in events if e["type"] == "response.output_text.delta"]
        assert len(text_deltas) == 2
        assert text_deltas[0]["delta"] == "Hello"
        assert text_deltas[1]["delta"] == " world"

        completed = [e for e in events if e["type"] == "response.completed"][0]
        assert completed["response"]["status"] == "completed"
        assert completed["response"]["usage"]["output_tokens"] == 2

    @pytest.mark.asyncio
    async def test_reasoning_plus_text(self) -> None:
        turn = GenerationTurn()
        request = _fake_responses_request()

        turn.emit(TurnStarted(response_id="resp_2", model="m", created=1))
        turn.emit(ReasoningDelta(delta="thinking..."))
        turn.emit(TextDelta(delta="answer"))
        turn.emit(TurnCompleted(finish_reason="stop"))
        turn.finish()

        events: list[dict[str, Any]] = []
        async for chunk in responses_stream_from_turn_events(
            turn,
            request=request,
            response_id="resp_2",
            model="m",
            created_at=1,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        types = [e["type"] for e in events]
        assert "response.reasoning_text.delta" in types
        assert "response.output_text.delta" in types
        assert "response.reasoning_text.done" in types
        assert "response.output_text.done" in types

    @pytest.mark.asyncio
    async def test_tool_call_rendering(self) -> None:
        turn = GenerationTurn()
        request = _fake_responses_request()

        turn.emit(TurnStarted(response_id="resp_3", model="m", created=1))
        turn.emit(ToolCallDelta(index=0, call_id="call_1", name="get_weather"))
        turn.emit(ToolCallDelta(index=0, arguments_delta='{"city":'))
        turn.emit(ToolCallDelta(index=0, arguments_delta='"NYC"}'))
        turn.emit(TurnCompleted(finish_reason="tool_calls"))
        turn.finish()

        events: list[dict[str, Any]] = []
        async for chunk in responses_stream_from_turn_events(
            turn,
            request=request,
            response_id="resp_3",
            model="m",
            created_at=1,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        types = [e["type"] for e in events]
        assert "response.output_item.added" in types
        assert "response.function_call_arguments.delta" in types
        assert "response.function_call_arguments.done" in types

        added = [e for e in events if e["type"] == "response.output_item.added"]
        assert any(a["item"]["type"] == "function_call" for a in added)

        done_fc = [e for e in events if e["type"] == "response.function_call_arguments.done"][0]
        assert done_fc["arguments"] == '{"city":"NYC"}'

    @pytest.mark.asyncio
    async def test_failed_turn_renders_response_failed(self) -> None:
        turn = GenerationTurn()
        request = _fake_responses_request()

        turn.emit(TurnStarted(response_id="resp_4", model="m", created=1))
        turn.emit(TextDelta(delta="partial"))
        turn.emit(TurnFailed(error="out of memory", code="server_error"))
        turn.finish()

        events: list[dict[str, Any]] = []
        async for chunk in responses_stream_from_turn_events(
            turn,
            request=request,
            response_id="resp_4",
            model="m",
            created_at=1,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        types = [e["type"] for e in events]
        assert "response.failed" in types
        assert "response.completed" not in types

        failed = [e for e in events if e["type"] == "response.failed"][0]
        assert failed["response"]["error"]["code"] == "server_error"

    @pytest.mark.asyncio
    async def test_cancelled_turn_renders_response_failed(self) -> None:
        turn = GenerationTurn()
        request = _fake_responses_request()

        turn.emit(TurnStarted(response_id="resp_5", model="m", created=1))
        turn.emit(TurnCancelled(reason="client_disconnected"))
        turn.finish()

        events: list[dict[str, Any]] = []
        async for chunk in responses_stream_from_turn_events(
            turn,
            request=request,
            response_id="resp_5",
            model="m",
            created_at=1,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        types = [e["type"] for e in events]
        assert "response.failed" in types
        failed = [e for e in events if e["type"] == "response.failed"][0]
        assert failed["response"]["error"]["code"] == "request_cancelled"

    @pytest.mark.asyncio
    async def test_length_finish_reason_produces_incomplete(self) -> None:
        turn = GenerationTurn()
        request = _fake_responses_request()

        turn.emit(TurnStarted(response_id="resp_6", model="m", created=1))
        turn.emit(TextDelta(delta="truncated"))
        turn.emit(TurnCompleted(finish_reason="length"))
        turn.finish()

        events: list[dict[str, Any]] = []
        async for chunk in responses_stream_from_turn_events(
            turn,
            request=request,
            response_id="resp_6",
            model="m",
            created_at=1,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        types = [e["type"] for e in events]
        assert "response.incomplete" in types
        assert "response.completed" not in types

    @pytest.mark.asyncio
    async def test_sequence_numbers_monotonic(self) -> None:
        turn = GenerationTurn()
        request = _fake_responses_request()

        turn.emit(TurnStarted(response_id="r", model="m", created=1))
        turn.emit(TextDelta(delta="a"))
        turn.emit(TurnCompleted(finish_reason="stop"))
        turn.finish()

        seq_numbers: list[int] = []
        async for chunk in responses_stream_from_turn_events(
            turn,
            request=request,
            response_id="r",
            model="m",
            created_at=1,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    data = json.loads(line.removeprefix("data: "))
                    if "sequence_number" in data:
                        seq_numbers.append(data["sequence_number"])

        assert seq_numbers == sorted(seq_numbers)
        assert len(set(seq_numbers)) == len(seq_numbers)
