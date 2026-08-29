"""Contract tests for the native, ephemeral Responses adapter."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import time

import pytest
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import create_app
from mtplx.server.protocols.responses import (
    ResponsesProtocolError,
    ResponsesRequest,
    response_from_chat_completion,
    responses_request_to_chat,
    responses_stream_from_chat_sse,
)
from test_server_openai import _fake_generation, _fake_state, _fake_streaming_generation


def _event_payloads(response_text: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response_text.splitlines()
        if line.startswith("data: {")
    ]


def _translated_events(chat_payloads: list[dict]) -> list[dict]:
    async def chat_frames():
        for payload in chat_payloads:
            yield f"data: {json.dumps(payload)}\n\n"

    async def collect():
        events: list[dict] = []
        async for frame in responses_stream_from_chat_sse(
            chat_frames(),
            request=ResponsesRequest(input="hi", stream=True),
            response_id="resp_failure",
            model="buddy",
            created_at=123,
        ):
            data_line = next(
                line for line in frame.splitlines() if line.startswith("data: ")
            )
            events.append(json.loads(data_line[6:]))
        return events

    return asyncio.run(collect())


def _ready_client(monkeypatch, *, text: str = "Hello") -> TestClient:
    state = _fake_state()
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(openai, "_run_generation", lambda *_a, **_kw: _fake_generation(text))
    return TestClient(create_app(state))


def test_request_translation_accepts_messages_controls_tools_and_images():
    request = ResponsesRequest.model_validate(
        {
            "model": "buddy",
            "instructions": "Be concise.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe it"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.test/cat.png",
                        },
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"q":"cat"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "found",
                },
            ],
            "max_output_tokens": 64,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 20,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "reasoning": {"effort": "medium"},
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look something up",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "function", "name": "lookup"},
        }
    )

    chat = responses_request_to_chat(request)

    assert chat["messages"][0] == {"role": "system", "content": "Be concise."}
    assert chat["messages"][1]["content"][0] == {
        "type": "text",
        "text": "Describe it",
    }
    assert chat["messages"][1]["content"][1]["image_url"]["url"].startswith("https:")
    assert chat["messages"][2]["tool_calls"][0]["id"] == "call_1"
    assert chat["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "found",
    }
    assert len(chat["messages"]) == 4
    assert chat["max_tokens"] == 64
    assert chat["reasoning_effort"] == "medium"
    assert chat["tools"][0]["function"]["name"] == "lookup"
    assert chat["tool_choice"] == {
        "type": "function",
        "function": {"name": "lookup"},
    }


def test_request_translation_keeps_parallel_tool_calls_in_one_burst():
    chat = responses_request_to_chat(
        ResponsesRequest.model_validate(
            {
                "input": [
                    {"role": "user", "content": "Check both sources"},
                    {"type": "function_call", "call_id": "call_search", "name": "web_search", "arguments": '{"query":"loctree"}'},
                    {"type": "function_call", "call_id": "call_image", "name": "inspect_image", "arguments": '{"url":"https://example.test/image.png"}'},
                    {"type": "function_call_output", "call_id": "call_search", "output": "search result"},
                    {"type": "function_call_output", "call_id": "call_image", "output": "image result"},
                ]
            }
        )
    )

    assert [message["role"] for message in chat["messages"]] == [
        "user", "assistant", "tool", "tool"
    ]
    assert [call["id"] for call in chat["messages"][1]["tool_calls"]] == [
        "call_search", "call_image"
    ]
    assert [
        message["tool_call_id"]
        for message in chat["messages"]
        if message["role"] == "tool"
    ] == ["call_search", "call_image"]


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"input": "hi", "store": True}, "store"),
        ({"input": "hi", "previous_response_id": "resp_old"}, "previous_response_id"),
        (
            {
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_file", "file_id": "file_1"}],
                    }
                ]
            },
            "input[0].content[0]",
        ),
        (
            {"input": "hi", "text": {"format": {"type": "json_schema"}}},
            "text.format",
        ),
        ({"input": "hi", "text": {"verbosity": "high"}}, "text.verbosity"),
        (
            {"input": "hi", "reasoning": {"summary": "detailed"}},
            "reasoning.summary",
        ),
        ({"input": "hi", "background": True}, "background"),
        (
            {"input": "hi", "tools": [{"type": "web_search_preview"}]},
            "tools[0].type",
        ),
        (
            {"input": "hi", "tool_choice": {"type": "web_search_preview"}},
            "tool_choice.type",
        ),
    ],
)
def test_unsupported_semantics_fail_with_exact_parameter(body, param):
    with pytest.raises(ResponsesProtocolError) as raised:
        responses_request_to_chat(ResponsesRequest.model_validate(body))
    assert raised.value.param == param


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"input": "hi", "text": {"verbosity": "high"}}, "text.verbosity"),
        (
            {"input": "hi", "reasoning": {"summary": "detailed"}},
            "reasoning.summary",
        ),
    ],
)
def test_post_responses_rejects_unimplemented_controls(monkeypatch, body, param):
    response = _ready_client(monkeypatch).post("/v1/responses", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["param"] == param


def test_nonstream_encoder_separates_reasoning_text_and_function_calls():
    payload = response_from_chat_completion(
        {
            "model": "buddy",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "private chain",
                        "content": "Visible preamble.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"q":"cat"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "mtplx_stats": {"generation_mode": "mtp", "mtp_depth": 3},
        },
        request=ResponsesRequest(input="hi"),
        response_id="resp_test",
        created_at=123,
    )

    assert [item["type"] for item in payload["output"]] == [
        "reasoning",
        "message",
        "function_call",
    ]
    assert payload["output"][0]["content"][0]["text"] == "private chain"
    assert payload["output"][1]["content"][0]["text"] == "Visible preamble."
    assert payload["output"][2]["arguments"] == '{"q":"cat"}'
    assert payload["usage"]["input_tokens"] == 3
    assert payload["store"] is False


def test_post_responses_nonstream_calls_existing_generation_once(monkeypatch):
    calls = 0
    state = _fake_state()
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])

    def generate(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _fake_generation("Native response")

    monkeypatch.setattr(openai, "_run_generation", generate)
    response = TestClient(create_app(state)).post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "model": "buddy",
            "instructions": "Answer directly.",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "store": False,
            "max_output_tokens": 16,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert calls == 1
    assert payload["object"] == "response"
    assert payload["status"] == "completed"
    assert payload["output"][0]["content"][0]["text"] == "Native response"
    assert payload["instructions"] == "Answer directly."


def test_post_responses_id_shares_validated_request_hint(monkeypatch):
    response = _ready_client(monkeypatch).post(
        "/v1/responses",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-request-id": "trace123",
        },
        json={"input": "Hi"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["id"] == "resp-trace123"


def test_post_responses_stream_orders_native_events_and_sequences(monkeypatch):
    state = _fake_state()
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation("Hello"))
    response = TestClient(create_app(state)).post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={"input": "Hi", "stream": True, "max_output_tokens": 16},
    )

    assert response.status_code == 200, response.text
    events = _event_payloads(response.text)
    event_types = [event["type"] for event in events]
    assert event_types[:2] == ["response.created", "response.in_progress"]
    assert "response.output_item.added" in event_types
    assert "response.content_part.added" in event_types
    assert "response.output_text.delta" in event_types
    assert event_types[-1] == "response.completed"
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert events[0]["response"]["model"] == state.model_id
    assert events[-1]["response"]["model"] == state.model_id
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Hello"


def test_post_responses_stream_keeps_long_live_generation_observable(monkeypatch):
    state = _fake_state()
    state.args.stats_footer = False
    state.generation_executor = ThreadPoolExecutor(max_workers=1)
    tokens = [ord("o"), ord("k")]

    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(openai, "STREAM_HEARTBEAT_INTERVAL_S", 0.0)
    monkeypatch.setattr(openai, "STREAM_SILENCE_WARN_S", 0.01)
    monkeypatch.setattr(openai, "STREAM_SILENCE_WARN_INTERVAL_S", 60.0)

    def generate(_state, _prompt_ids, **kwargs):
        time.sleep(1.25)
        kwargs["token_callback"](tokens)
        return {
            "text": "ok",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": 3,
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", generate)

    try:
        response = TestClient(create_app(state)).post(
            "/v1/responses",
            headers={
                "x-mtplx-cache-mode": "bypass",
                "x-mtplx-allow-client-controls": "1",
            },
            json={
                "input": "Say ok.",
                "stream": True,
                "max_output_tokens": 16,
            },
        )
    finally:
        state.generation_executor.shutdown(wait=True)

    assert response.status_code == 200, response.text
    assert ": mtplx-heartbeat\n\n" in response.text
    events = _event_payloads(response.text)
    assert [
        event["delta"]
        for event in events
        if event["type"] == "response.output_text.delta"
    ] == ["ok"]
    assert events[-1]["type"] == "response.completed"


def test_post_responses_stream_separates_reasoning_text_and_function_call(monkeypatch):
    state = _fake_state()
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "<think>Private plan</think>\n"
            "Visible preamble.\n"
            "<tool_call>\n"
            "<function=lookup>\n"
            "<parameter=q>\ncat\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    )
    response = TestClient(create_app(state)).post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "input": "Use the tool",
            "stream": True,
            "max_output_tokens": 128,
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    events = _event_payloads(response.text)
    reasoning = "".join(
        event["delta"]
        for event in events
        if event["type"] == "response.reasoning_text.delta"
    )
    visible = "".join(
        event["delta"]
        for event in events
        if event["type"] == "response.output_text.delta"
    )
    argument_deltas = [
        event
        for event in events
        if event["type"] == "response.function_call_arguments.delta"
    ]
    argument_done = next(
        event
        for event in events
        if event["type"] == "response.function_call_arguments.done"
    )
    completed = events[-1]["response"]

    assert reasoning == "Private plan"
    assert reasoning not in visible
    assert visible.strip() == "Visible preamble."
    assert "".join(event["delta"] for event in argument_deltas) == '{"q":"cat"}'
    assert argument_done["name"] == "lookup"
    assert argument_done["arguments"] == '{"q":"cat"}'
    assert [item["type"] for item in completed["output"]] == [
        "reasoning",
        "message",
        "function_call",
    ]
    assert completed["output"][0]["content"][0]["text"] == "Private plan"
    assert completed["output"][1]["content"][0]["text"].strip() == "Visible preamble."
    assert completed["output"][2]["name"] == "lookup"
    assert completed["output"][2]["arguments"] == '{"q":"cat"}'


def test_stream_translator_delivers_first_delta_before_completion():
    async def chat_frames():
        yield 'data: {"model":"buddy","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        await asyncio.sleep(0.02)
        yield 'data: {"choices":[{"delta":{"content":"early"},"finish_reason":null}]}\n\n'
        await asyncio.sleep(0.08)
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
        yield "data: [DONE]\n\n"

    async def collect():
        observed: list[tuple[float, dict]] = []
        started = time.monotonic()
        async for frame in responses_stream_from_chat_sse(
            chat_frames(),
            request=ResponsesRequest(input="hi", stream=True),
            response_id="resp_temporal",
            model="buddy",
            created_at=123,
        ):
            data_line = next(line for line in frame.splitlines() if line.startswith("data: "))
            observed.append((time.monotonic() - started, json.loads(data_line[6:])))
        return observed

    observed = asyncio.run(collect())
    delta_time = next(at for at, event in observed if event["type"] == "response.output_text.delta")
    completed_time = next(at for at, event in observed if event["type"] == "response.completed")
    assert delta_time < 0.07
    assert completed_time >= 0.09
    assert delta_time < completed_time


def test_stream_translator_preserves_bursts_and_idle_heartbeats():
    async def chat_frames():
        yield 'data: {"choices":[{"delta":{"content":"first"},"finish_reason":null}]}\n\n'
        await asyncio.sleep(0.02)
        yield 'data: {"mtplx_progress":{"heartbeat":true,"phase":"generating"}}\n\n'
        await asyncio.sleep(0.02)
        yield 'data: {"choices":[{"delta":{"content":" second"},"finish_reason":null}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    async def collect():
        frames: list[str] = []
        async for frame in responses_stream_from_chat_sse(
            chat_frames(),
            request=ResponsesRequest(input="hi", stream=True),
            response_id="resp_bursts",
            model="buddy",
            created_at=123,
        ):
            frames.append(frame)
        return frames

    frames = asyncio.run(collect())
    heartbeat_index = frames.index(": mtplx-heartbeat\n\n")
    events = [
        json.loads(line[6:])
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("data: {")
    ]
    deltas = [event["delta"] for event in events if event["type"] == "response.output_text.delta"]
    delta_indices = [
        index
        for index, frame in enumerate(frames)
        if '"type": "response.output_text.delta"' in frame
    ]

    assert deltas == ["first", " second"]
    assert delta_indices[0] < heartbeat_index < delta_indices[1]
    assert events[-1]["type"] == "response.completed"


def test_stream_translator_keeps_item_indices_stable_in_arrival_order():
    async def chat_frames():
        yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{\\"q\\":"}}]},"finish_reason":null}]}\n\n'
        yield 'data: {"choices":[{"delta":{"content":"Visible"},"finish_reason":null}]}\n\n'
        yield 'data: {"choices":[{"delta":{"reasoning_content":"Private"},"finish_reason":null}]}\n\n'
        yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_2","function":{"name":"save","arguments":"{}"}},{"index":0,"function":{"arguments":"\\"cat\\"}"}}]},"finish_reason":null}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
        yield "data: [DONE]\n\n"

    async def collect():
        events: list[dict] = []
        async for frame in responses_stream_from_chat_sse(
            chat_frames(),
            request=ResponsesRequest(input="hi", stream=True),
            response_id="resp_reordered",
            model="buddy",
            created_at=123,
        ):
            data_line = next(line for line in frame.splitlines() if line.startswith("data: "))
            events.append(json.loads(data_line[6:]))
        return events

    events = asyncio.run(collect())
    added = [event for event in events if event["type"] == "response.output_item.added"]
    assert [(event["item"]["type"], event["output_index"]) for event in added] == [
        ("function_call", 0),
        ("message", 1),
        ("reasoning", 2),
        ("function_call", 3),
    ]
    argument_deltas = [
        event
        for event in events
        if event["type"] == "response.function_call_arguments.delta"
    ]
    assert [(event["output_index"], event["delta"]) for event in argument_deltas] == [
        (0, '{"q":'),
        (3, "{}"),
        (0, '"cat"}'),
    ]

    indices_by_item = {event["item"]["id"]: event["output_index"] for event in added}
    for event in events:
        item_id = event.get("item_id")
        if item_id in indices_by_item:
            assert event["output_index"] == indices_by_item[item_id]
        item = event.get("item")
        if isinstance(item, dict) and item.get("id") in indices_by_item:
            assert event["output_index"] == indices_by_item[item["id"]]

    completed = events[-1]
    assert completed["type"] == "response.completed"
    assert [item["type"] for item in completed["response"]["output"]] == [
        "function_call",
        "message",
        "reasoning",
        "function_call",
    ]
    assert completed["response"]["output"][0]["arguments"] == '{"q":"cat"}'
    argument_done = [
        event
        for event in events
        if event["type"] == "response.function_call_arguments.done"
    ]
    assert [event["output_index"] for event in argument_done] == [0, 3]
    assert [event["name"] for event in argument_done] == ["lookup", "save"]


def test_stream_translator_normalizes_failure_before_output():
    events = _translated_events(
        [
            {
                "choices": [{"delta": {}, "finish_reason": "error"}],
                "error": {
                    "message": "generation failed",
                    "type": "server_error",
                    "code": "RuntimeError",
                    "param": None,
                },
            }
        ]
    )

    failed = events[-1]
    assert failed["type"] == "response.failed"
    assert failed["response"]["status"] == "failed"
    assert failed["response"]["output"] == []
    assert failed["response"]["error"] == {
        "code": "server_error",
        "message": "generation failed",
    }


def test_stream_translator_marks_partial_outputs_incomplete_on_failure():
    events = _translated_events(
        [
            {
                "choices": [
                    {
                        "delta": {"reasoning_content": "Private"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {"delta": {"content": "Visible"}, "finish_reason": None}
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"q":',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [{"delta": {}, "finish_reason": "error"}],
                "error": {
                    "message": "generation failed after partial output",
                    "type": "server_error",
                    "code": "RuntimeError",
                    "param": None,
                },
            },
        ]
    )

    failed = events[-1]
    assert failed["type"] == "response.failed"
    assert [item["type"] for item in failed["response"]["output"]] == [
        "reasoning",
        "message",
        "function_call",
    ]
    assert {item["status"] for item in failed["response"]["output"]} == {
        "incomplete"
    }
    assert failed["response"]["output"][2]["arguments"] == '{"q":'
    assert failed["response"]["error"]["code"] == "server_error"


def test_stream_translator_normalizes_server_side_cancellation_failure():
    events = _translated_events(
        [
            {
                "choices": [{"delta": {}, "finish_reason": "error"}],
                "error": {
                    "message": "request cancelled",
                    "type": "server_error",
                    "code": "CancelledError",
                    "param": None,
                },
            }
        ]
    )

    failed = events[-1]
    assert failed["type"] == "response.failed"
    assert failed["response"]["error"] == {
        "code": "server_error",
        "message": "request cancelled",
    }


def test_stateful_response_lifecycle_fails_explicitly(monkeypatch):
    client = _ready_client(monkeypatch)
    create = client.post("/v1/responses", json={"input": "hi", "store": True})
    assert create.status_code == 400
    assert create.json()["error"]["param"] == "store"
    for response in (
        client.get("/v1/responses/resp_missing"),
        client.delete("/v1/responses/resp_missing"),
        client.post("/v1/responses/resp_missing/cancel"),
    ):
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "not_implemented"
