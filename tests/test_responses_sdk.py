"""The mounted native adapter's wire contract against the CI-pinned SDK."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

pytest.importorskip("openai")
from openai.types.responses import Response, ResponseStreamEvent
from test_responses_api import _event_payloads, _ready_client, _translated_events
from test_server_openai import _fake_streaming_generation

from mtplx.server import openai


def test_native_json_and_stored_response_validate_with_sdk(monkeypatch):
    client = _ready_client(monkeypatch)
    first = client.post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={"input": "Hi", "store": True},
    )
    assert first.status_code == 200, first.text
    parsed = TypeAdapter(Response).validate_python(first.json(), strict=True)
    assert parsed.status == "completed"
    stored = client.get(f"/v1/responses/{parsed.id}")
    assert stored.status_code == 200
    TypeAdapter(Response).validate_python(stored.json(), strict=True)


@pytest.mark.parametrize(
    "text",
    [
        "Hello",
        "<think>Private plan</think>Visible preamble.<tool_call>\n<function=lookup>\n<parameter=q>\ncat\n</parameter>\n</function>\n</tool_call>",
    ],
)
def test_native_sse_events_validate_with_sdk(monkeypatch, text):
    client = _ready_client(monkeypatch)
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation(text))
    reply = client.post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "input": "Hi",
            "stream": True,
            "max_output_tokens": 128,
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                }
            ],
        },
    )
    assert reply.status_code == 200, reply.text
    events = _event_payloads(reply.text)
    for event in events:
        TypeAdapter(ResponseStreamEvent).validate_python(event, strict=True)
    assert events[-1]["type"] == "response.completed"
    if "Private plan" in text:
        assert any(e["type"] == "response.function_call_arguments.done" for e in events)
        assert any(e["type"] == "response.reasoning_text.delta" for e in events)


def test_partial_failure_events_validate_with_sdk():
    events = _translated_events(
        [
            {
                "choices": [
                    {"index": 0, "delta": {"content": "Partial"}, "finish_reason": None}
                ]
            },
            {"error": {"message": "generation failed", "code": "generation_error"}},
        ]
    )
    for event in events:
        TypeAdapter(ResponseStreamEvent).validate_python(event, strict=True)
    assert events[-1]["type"] == "response.failed"
