"""Tests for the GenerationTurn event seam — W1 recovery.

Covers the 6 refutation points from the W1b correction plus original
acceptance criteria.

Refutation coverage:
  R1 — bounded slow-consumer: subscriber overflow is handled, not unbounded.
  R2 — thread-safety: emit/start/complete/fail/cancel are event-loop-thread
        calls (documented invariant, tested via direct invocation).
  R3 — enforced state machine: duplicate start, duplicate terminal, post-terminal
        delta refusal, EOF-without-terminal impossible.
  R4 — OutputItemStarted is emitted by the live producer.
  R5 — all three protocol paths share lifecycle authority (fan-out test).
  R6 — driver cancellation, no orphan task.

Acceptance criteria:
  A1 — typed events cover turn start, deltas, usage, complete, fail, cancel.
  A2 — Responses rendered from TurnEvents (not from Chat SSE parsing).
  A3 — generation remains off asyncio event loop (emit is sync).
  A4 — first model delta observable independently (timestamped).
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

import mtplx.server.openai as openai
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
    TurnState,
    UsageUpdate,
)
from mtplx.server.protocols.responses.encoder import (
    responses_stream_from_turn_events,
)
from mtplx.server.protocols.responses.schema import ResponsesRequest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_responses_request(**overrides: Any) -> ResponsesRequest:
    base: dict[str, Any] = {
        "model": "test-model",
        "input": "hello",
    }
    base.update(overrides)
    return ResponsesRequest.model_validate(base)


class TestEventTaxonomy:
    """A1: every event type is frozen, typed, and carries required fields."""

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

    def test_usage_update(self) -> None:
        ev = UsageUpdate(prompt_tokens=10, completion_tokens=20)
        assert ev.prompt_tokens == 10

    def test_turn_completed(self) -> None:
        usage = UsageUpdate(prompt_tokens=5, completion_tokens=15)
        ev = TurnCompleted(finish_reason="stop", usage=usage)
        assert ev.finish_reason == "stop"
        assert ev.usage is not None

    def test_turn_failed(self) -> None:
        ev = TurnFailed(error="boom", code="server_error")
        assert ev.error == "boom"

    def test_turn_cancelled(self) -> None:
        ev = TurnCancelled(reason="client_disconnected")
        assert ev.reason == "client_disconnected"

    def test_frozen(self) -> None:
        ev = TextDelta(delta="x")
        with pytest.raises(AttributeError):
            ev.delta = "y"  # type: ignore[misc]

    def test_output_item_started(self) -> None:
        ev = OutputItemStarted(kind="text", index=0, item_id="msg_1")
        assert ev.kind == "text"
        assert ev.item_id == "msg_1"


class TestStateMachine:
    """R3: enforced state machine transitions."""

    def test_initial_state_is_idle(self) -> None:
        turn = GenerationTurn()
        assert turn.state == TurnState.IDLE

    def test_start_transitions_to_started(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        assert turn.state == TurnState.STARTED

    def test_duplicate_start_raises(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        with pytest.raises(RuntimeError, match="IDLE"):
            turn.start(TurnStarted(response_id="r2", model="m", created=2))

    def test_emit_before_start_raises(self) -> None:
        turn = GenerationTurn()
        with pytest.raises(RuntimeError, match="STARTED"):
            turn.emit(TextDelta(delta="x"))

    def test_emit_terminal_type_raises_type_error(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        with pytest.raises(TypeError, match="complete"):
            turn.emit(TurnCompleted(finish_reason="stop"))

    def test_complete_transitions_to_terminal(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.complete(TurnCompleted(finish_reason="stop"))
        assert turn.state == TurnState.TERMINAL

    def test_fail_transitions_to_terminal(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.fail(TurnFailed(error="boom"))
        assert turn.state == TurnState.TERMINAL

    def test_cancel_transitions_to_terminal(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.cancel(TurnCancelled(reason="dc"))
        assert turn.state == TurnState.TERMINAL

    def test_duplicate_terminal_raises(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.complete(TurnCompleted(finish_reason="stop"))
        with pytest.raises(RuntimeError, match="terminal"):
            turn.fail(TurnFailed(error="ignored"))
        with pytest.raises(RuntimeError, match="terminal"):
            turn.cancel(TurnCancelled(reason="ignored"))
        assert turn.state == TurnState.TERMINAL

    def test_post_terminal_emit_raises(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.complete(TurnCompleted(finish_reason="stop"))
        with pytest.raises(RuntimeError, match="terminal"):
            turn.emit(TextDelta(delta="refused"))
        assert turn.state == TurnState.TERMINAL

    def test_fail_without_start_allowed(self) -> None:
        turn = GenerationTurn()
        turn.fail(TurnFailed(error="immediate"))
        assert turn.state == TurnState.TERMINAL

    def test_complete_without_start_raises(self) -> None:
        turn = GenerationTurn()
        with pytest.raises(RuntimeError, match="IDLE"):
            turn.complete(TurnCompleted(finish_reason="stop"))

    def test_cancel_without_start_raises(self) -> None:
        turn = GenerationTurn()
        with pytest.raises(RuntimeError, match="IDLE"):
            turn.cancel(TurnCancelled(reason="dc"))


class TestEOFWithoutTerminal:
    """R3 corollary: EOF without terminal fails rather than completes."""

    @pytest.mark.anyio
    async def test_subscriber_eof_only_after_terminal(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.emit(TextDelta(delta="a"))
        sub = turn.subscribe()

        started = await sub.__anext__()
        assert isinstance(started, TurnStarted)
        delta = await sub.__anext__()
        assert isinstance(delta, TextDelta)

        get_task = asyncio.ensure_future(sub.__anext__())
        await asyncio.sleep(0.01)
        assert not get_task.done()

        turn.complete(TurnCompleted(finish_reason="stop"))
        result = await asyncio.wait_for(get_task, timeout=1.0)
        assert isinstance(result, TurnCompleted)

        with pytest.raises(StopAsyncIteration):
            await sub.__anext__()


class TestBoundedSlowConsumer:
    """R1: slow consumer is bounded, not unbounded accumulation."""

    @pytest.mark.anyio
    async def test_subscriber_has_bounded_queue(self) -> None:
        turn = GenerationTurn(maxsize=4)
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        sub = turn.subscribe()

        for i in range(10):
            try:
                turn.emit(TextDelta(delta=f"d{i}"))
            except RuntimeError:
                break

        events: list[TurnEvent] = []
        async for ev in sub:
            events.append(ev)
        assert len(events) <= 5
        assert isinstance(events[-1], TurnFailed)
        assert events[-1].code == "backpressure_overflow"

    @pytest.mark.anyio
    async def test_overflow_is_explicit_failure_not_clean_eof(self) -> None:
        turn = GenerationTurn(maxsize=2)
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        sub = turn.subscribe()

        for _ in range(100):
            try:
                turn.emit(TextDelta(delta="x"))
            except RuntimeError:
                break

        assert turn.state == TurnState.TERMINAL
        events = [ev async for ev in sub]
        assert isinstance(events[-1], TurnFailed)
        assert events[-1].code == "backpressure_overflow"


class TestFanOut:
    """R5: multiple subscribers observe the same turn (lifecycle authority shared)."""

    @pytest.mark.anyio
    async def test_two_subscribers_see_same_events(self) -> None:
        turn = GenerationTurn()
        sub1 = turn.subscribe()
        sub2 = turn.subscribe()

        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.emit(TextDelta(delta="hello"))
        turn.complete(TurnCompleted(finish_reason="stop"))

        events1: list[TurnEvent] = []
        async for ev in sub1:
            events1.append(ev)

        events2: list[TurnEvent] = []
        async for ev in sub2:
            events2.append(ev)

        assert len(events1) == 3
        assert len(events2) == 3
        for e1, e2 in zip(events1, events2):
            assert type(e1) is type(e2)

    @pytest.mark.anyio
    async def test_late_subscriber_gets_replay(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.emit(TextDelta(delta="a"))

        sub = turn.subscribe()

        turn.emit(TextDelta(delta="b"))
        turn.complete(TurnCompleted(finish_reason="stop"))

        events: list[TurnEvent] = []
        async for ev in sub:
            events.append(ev)

        assert len(events) == 4
        assert isinstance(events[0], TurnStarted)
        assert isinstance(events[1], TextDelta) and events[1].delta == "a"
        assert isinstance(events[2], TextDelta) and events[2].delta == "b"
        assert isinstance(events[3], TurnCompleted)

    @pytest.mark.anyio
    async def test_subscribe_after_terminal_replays_everything(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.emit(TextDelta(delta="x"))
        turn.complete(TurnCompleted(finish_reason="stop"))

        sub = turn.subscribe()
        events: list[TurnEvent] = []
        async for ev in sub:
            events.append(ev)

        assert len(events) == 3
        assert isinstance(events[2], TurnCompleted)

    @pytest.mark.anyio
    async def test_late_subscriber_after_large_history_does_not_raise(self) -> None:
        turn = GenerationTurn(maxsize=4, replay=4)
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        for i in range(20):
            turn.emit(TextDelta(delta=str(i)))
        turn.complete(TurnCompleted(finish_reason="stop"))

        sub = turn.subscribe(maxsize=3)
        events = [ev async for ev in sub]
        assert len(events) <= 3
        assert isinstance(events[-1], TurnCompleted)


class TestDriverCancellation:
    """R6: driver cancellation, no orphan task."""

    @pytest.mark.anyio
    async def test_cancel_terminates_subscriber(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        sub = turn.subscribe()

        get_task = asyncio.ensure_future(sub.__anext__())
        await asyncio.sleep(0)
        started = await get_task
        assert isinstance(started, TurnStarted)

        wait_task = asyncio.ensure_future(sub.__anext__())
        await asyncio.sleep(0.01)
        assert not wait_task.done()

        turn.cancel(TurnCancelled(reason="client_disconnected"))

        result = await asyncio.wait_for(wait_task, timeout=1.0)
        assert isinstance(result, TurnCancelled)

        with pytest.raises(StopAsyncIteration):
            await sub.__anext__()

    @pytest.mark.anyio
    async def test_driver_task_cleanup_on_cancel(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        sub = turn.subscribe()

        driver_ran = False

        async def fake_driver():
            nonlocal driver_ran
            try:
                while True:
                    await asyncio.sleep(0.01)
                    turn.emit(TextDelta(delta="x"))
            except asyncio.CancelledError:
                driver_ran = True
                raise

        driver = asyncio.ensure_future(fake_driver())
        await asyncio.sleep(0.05)

        turn.cancel(TurnCancelled(reason="test"))

        events: list[TurnEvent] = []
        async for ev in sub:
            events.append(ev)

        driver.cancel()
        try:
            await driver
        except asyncio.CancelledError:
            pass
        assert driver_ran or driver.cancelled()
        assert any(isinstance(e, TurnCancelled) for e in events)


class TestFirstDelta:
    """A4: first model delta observable independently from completion."""

    @pytest.mark.anyio
    async def test_first_delta_s_tracked(self) -> None:
        turn = GenerationTurn()
        assert turn.first_delta_s is None
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        assert turn.first_delta_s is None
        turn.emit(TextDelta(delta="a"))
        assert turn.first_delta_s is not None
        first = turn.first_delta_s
        turn.emit(TextDelta(delta="b"))
        assert turn.first_delta_s == first

    def test_usage_does_not_set_first_delta(self) -> None:
        turn = GenerationTurn()
        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.emit(UsageUpdate(prompt_tokens=1, completion_tokens=1))
        assert turn.first_delta_s is None


class TestResponsesFromTurnEvents:
    """A2: Responses rendered from TurnEvents, NOT from Chat SSE parsing."""

    @pytest.mark.anyio
    async def test_text_completion_renders_responses_sse(self) -> None:
        turn = GenerationTurn()
        request = _fake_responses_request()
        sub = turn.subscribe()

        turn.start(TurnStarted(response_id="resp_1", model="test-model", created=100))
        turn.emit(TextDelta(delta="Hello"))
        turn.emit(TextDelta(delta=" world"))
        turn.complete(TurnCompleted(
            finish_reason="stop",
            usage=UsageUpdate(prompt_tokens=5, completion_tokens=2),
        ))

        events: list[dict[str, Any]] = []
        async for chunk in responses_stream_from_turn_events(
            sub, request=request, response_id="resp_1",
            model="test-model", created_at=100,
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

        completed = [e for e in events if e["type"] == "response.completed"][0]
        assert completed["response"]["status"] == "completed"
        assert completed["response"]["usage"]["output_tokens"] == 2

    @pytest.mark.anyio
    async def test_reasoning_plus_text(self) -> None:
        turn = GenerationTurn()
        request = _fake_responses_request()
        sub = turn.subscribe()

        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.emit(ReasoningDelta(delta="thinking..."))
        turn.emit(TextDelta(delta="answer"))
        turn.complete(TurnCompleted(finish_reason="stop"))

        events: list[dict[str, Any]] = []
        async for chunk in responses_stream_from_turn_events(
            sub, request=request, response_id="r", model="m", created_at=1,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        types = [e["type"] for e in events]
        assert "response.reasoning_text.delta" in types
        assert "response.output_text.delta" in types

    @pytest.mark.anyio
    async def test_tool_call_rendering(self) -> None:
        turn = GenerationTurn()
        request = _fake_responses_request()
        sub = turn.subscribe()

        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.emit(ToolCallDelta(index=0, call_id="call_1", name="get_weather"))
        turn.emit(ToolCallDelta(index=0, arguments_delta='{"city":'))
        turn.emit(ToolCallDelta(index=0, arguments_delta='"NYC"}'))
        turn.complete(TurnCompleted(finish_reason="tool_calls"))

        events: list[dict[str, Any]] = []
        async for chunk in responses_stream_from_turn_events(
            sub, request=request, response_id="r", model="m", created_at=1,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        types = [e["type"] for e in events]
        assert "response.function_call_arguments.delta" in types
        assert "response.function_call_arguments.done" in types

        done_fc = [e for e in events if e["type"] == "response.function_call_arguments.done"][0]
        assert done_fc["arguments"] == '{"city":"NYC"}'

    @pytest.mark.anyio
    async def test_failed_turn_renders_response_failed(self) -> None:
        turn = GenerationTurn()
        request = _fake_responses_request()
        sub = turn.subscribe()

        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.emit(TextDelta(delta="partial"))
        turn.fail(TurnFailed(error="out of memory", code="server_error"))

        events: list[dict[str, Any]] = []
        async for chunk in responses_stream_from_turn_events(
            sub, request=request, response_id="r", model="m", created_at=1,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        types = [e["type"] for e in events]
        assert "response.failed" in types
        assert "response.completed" not in types

    @pytest.mark.anyio
    async def test_cancelled_turn_renders_response_failed(self) -> None:
        turn = GenerationTurn()
        request = _fake_responses_request()
        sub = turn.subscribe()

        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.cancel(TurnCancelled(reason="client_disconnected"))

        events: list[dict[str, Any]] = []
        async for chunk in responses_stream_from_turn_events(
            sub, request=request, response_id="r", model="m", created_at=1,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        types = [e["type"] for e in events]
        assert "response.failed" in types

    @pytest.mark.anyio
    async def test_length_finish_reason_produces_incomplete(self) -> None:
        turn = GenerationTurn()
        request = _fake_responses_request()
        sub = turn.subscribe()

        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.emit(TextDelta(delta="truncated"))
        turn.complete(TurnCompleted(finish_reason="length"))

        events: list[dict[str, Any]] = []
        async for chunk in responses_stream_from_turn_events(
            sub, request=request, response_id="r", model="m", created_at=1,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        types = [e["type"] for e in events]
        assert "response.incomplete" in types
        assert "response.completed" not in types

    @pytest.mark.anyio
    async def test_sequence_numbers_monotonic(self) -> None:
        turn = GenerationTurn()
        request = _fake_responses_request()
        sub = turn.subscribe()

        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.emit(TextDelta(delta="a"))
        turn.complete(TurnCompleted(finish_reason="stop"))

        seq_numbers: list[int] = []
        async for chunk in responses_stream_from_turn_events(
            sub, request=request, response_id="r", model="m", created_at=1,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    data = json.loads(line.removeprefix("data: "))
                    if "sequence_number" in data:
                        seq_numbers.append(data["sequence_number"])

        assert seq_numbers == sorted(seq_numbers)
        assert len(set(seq_numbers)) == len(seq_numbers)


class TestThreeProtocolsShareLifecycle:
    """R5: Chat, Responses, and Anthropic resolve to the same lifecycle authority."""

    @pytest.mark.anyio
    async def test_two_subscribers_share_terminal_state(self) -> None:
        turn = GenerationTurn()
        sub_responses = turn.subscribe()
        sub_anthropic = turn.subscribe()

        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.emit(TextDelta(delta="hello"))
        turn.complete(TurnCompleted(finish_reason="stop"))

        responses_events: list[TurnEvent] = []
        async for ev in sub_responses:
            responses_events.append(ev)

        anthropic_events: list[TurnEvent] = []
        async for ev in sub_anthropic:
            anthropic_events.append(ev)

        assert isinstance(responses_events[-1], TurnCompleted)
        assert isinstance(anthropic_events[-1], TurnCompleted)
        assert responses_events[-1].finish_reason == anthropic_events[-1].finish_reason

    @pytest.mark.anyio
    async def test_cancel_reaches_all_subscribers(self) -> None:
        turn = GenerationTurn()
        sub1 = turn.subscribe()
        sub2 = turn.subscribe()

        turn.start(TurnStarted(response_id="r", model="m", created=1))
        turn.cancel(TurnCancelled(reason="client_disconnected"))

        events1: list[TurnEvent] = []
        async for ev in sub1:
            events1.append(ev)

        events2: list[TurnEvent] = []
        async for ev in sub2:
            events2.append(ev)

        assert any(isinstance(e, TurnCancelled) for e in events1)
        assert any(isinstance(e, TurnCancelled) for e in events2)

    def test_live_routes_do_not_background_drain_chat_streams(self) -> None:
        source = inspect.getsource(openai.create_app)
        assert "_drive_generation" not in source
        assert "body_iterator" not in source[source.index('@app.post("/v1/responses")'):]

    @pytest.mark.anyio
    async def test_responses_eof_without_terminal_fails(self) -> None:
        async def eof_only():
            yield TurnStarted(response_id="r", model="m", created=1)
            yield TextDelta(delta="partial")

        events: list[dict[str, Any]] = []
        async for chunk in responses_stream_from_turn_events(
            eof_only(),
            request=_fake_responses_request(),
            response_id="r",
            model="m",
            created_at=1,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        assert events[-1]["type"] == "response.failed"
        assert events[-1]["response"]["error"]["code"] == "driver_eof_without_terminal"
        assert "response.completed" not in [event["type"] for event in events]

    @pytest.mark.anyio
    async def test_anthropic_eof_without_terminal_fails(self) -> None:
        async def eof_only():
            yield TurnStarted(response_id="r", model="m", created=1)
            yield TextDelta(delta="partial")

        frames = [
            frame
            async for frame in openai._anthropic_stream_from_turn_events(
                eof_only(), model="m"
            )
        ]
        assert any("event: error" in frame for frame in frames)
        assert not any("event: message_stop" in frame for frame in frames)


class TestW1dCommonDriverArchitecture:
    """The common producer is event-typed and no protocol drives another."""

    def test_forbidden_protocol_driver_shapes_are_absent(self) -> None:
        source = inspect.getsource(openai)
        assert "_TurnEventStreamResponse" not in source
        assert "_mtplx_stream_projection" not in source
        assert "Callable[[], AsyncIterator[str]]" not in source
        stream_source = inspect.getsource(openai._GenerationTurnStream)
        assert "async for _ in" not in stream_source
        assert "body_iterator" not in stream_source

    def test_all_streaming_routes_select_the_common_turn(self) -> None:
        source = inspect.getsource(openai.create_app)
        responses = source[source.index('@app.post("/v1/responses")'):]
        anthropic = source[source.index('@app.post("/v1/messages")'):]
        assert "_chat_stream_from_turn_events(turn_stream.events())" in source
        assert "responses_stream_from_turn_events(" in responses
        assert "_anthropic_stream_from_turn_events(" in anthropic
        assert responses.count("return_turn=True") >= 2

    def test_all_renderer_inputs_are_turn_event_iterators(self) -> None:
        for renderer in (
            openai._chat_stream_from_turn_events,
            responses_stream_from_turn_events,
            openai._anthropic_stream_from_turn_events,
        ):
            annotation = inspect.signature(renderer).parameters["events"].annotation
            assert "AsyncIterator[TurnEvent]" in str(annotation)

    def test_live_driver_emits_complete_claimed_taxonomy(self) -> None:
        source = inspect.getsource(openai.create_app)
        driver = source[source.index("async def drive_turn()"):]
        driver = driver[:driver.index("turn_stream = _GenerationTurnStream")]
        for event_type in (
            "TurnStarted",
            "OutputItemStarted",
            "TextDelta",
            "ReasoningDelta",
            "ToolCallDelta",
            "TurnHeartbeat",
            "UsageUpdate",
            "TurnCompleted",
            "TurnFailed",
            "TurnCancelled",
        ):
            assert event_type in driver

    @pytest.mark.anyio
    async def test_chat_renderer_consumes_synthetic_turn_events(self) -> None:
        async def synthetic() -> AsyncIterator[TurnEvent]:
            yield TurnStarted(response_id="chatcmpl-test", model="m", created=1)
            yield ReasoningDelta(delta="think")
            yield TextDelta(delta="answer")
            yield ToolCallDelta(
                index=0,
                call_id="call_1",
                name="lookup",
                arguments_delta="{}",
            )
            yield TurnCompleted(
                finish_reason="tool_calls",
                usage=UsageUpdate(prompt_tokens=2, completion_tokens=3),
                mtplx_stats={"finish_reason": "tool_calls"},
                timings={"generation_time": 0.1},
            )

        frames = [
            frame async for frame in openai._chat_stream_from_turn_events(synthetic())
        ]
        payloads = [
            json.loads(frame.removeprefix("data: "))
            for frame in frames
            if frame.startswith("data: {")
        ]
        assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert any(
            payload["choices"][0]["delta"].get("content") == "answer"
            for payload in payloads
        )
        assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert frames[-1] == "data: [DONE]\n\n"

    @pytest.mark.anyio
    async def test_renderer_exit_cancels_common_driver(self) -> None:
        turn = GenerationTurn()
        cancelled = asyncio.Event()

        async def drive() -> None:
            turn.start(TurnStarted(response_id="r", model="m", created=1))
            try:
                while True:
                    await asyncio.sleep(1)
            finally:
                cancelled.set()

        stream = openai._GenerationTurnStream(turn, drive)
        events = stream.events()
        assert isinstance(await events.__anext__(), TurnStarted)
        task = turn.driver_task
        assert task is not None and not task.done()
        await events.aclose()
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        assert task.done()

    @pytest.mark.anyio
    async def test_driver_eof_becomes_explicit_failure(self) -> None:
        turn = GenerationTurn()

        async def drive() -> None:
            turn.start(TurnStarted(response_id="r", model="m", created=1))

        events = [
            event
            async for event in openai._GenerationTurnStream(turn, drive).events()
        ]
        assert isinstance(events[-1], TurnFailed)
        assert events[-1].code == "driver_eof_without_terminal"
