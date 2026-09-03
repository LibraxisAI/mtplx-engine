from __future__ import annotations

from collections import Counter
import threading
import time

import pytest
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import create_app
from mtplx.server.protocols.responses import ResponseRegistry, ResponseStoreError
from test_server_openai import (
    _fake_generation,
    _fake_state,
    _fake_streaming_session_state,
)


def _envelope(response_id: str, text: str = "ok") -> dict:
    return {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": f"msg_{response_id}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


def test_registry_commit_get_and_copy_boundary():
    registry = ResponseRegistry()
    registry.begin(
        "resp_a",
        store=True,
        materialized_messages=[{"role": "user", "content": "hello"}],
        cancel=lambda: None,
    )
    source = _envelope("resp_a")
    registry.commit(
        "resp_a",
        source,
        materialized_messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"},
        ],
    )

    source["output"][0]["content"][0]["text"] = "mutated"
    fetched = registry.get("resp_a")
    assert fetched["output"][0]["content"][0]["text"] == "ok"
    fetched["status"] = "changed"
    assert registry.get("resp_a")["status"] == "completed"
    assert registry.stats()["entries"] == 1
    assert registry.stats()["bytes"] > 0


def test_registry_distinguishes_missing_deleted_evicted_and_in_progress():
    registry = ResponseRegistry(max_entries=1)
    registry.begin(
        "resp_live",
        store=True,
        materialized_messages=[],
        cancel=lambda: None,
    )
    with pytest.raises(ResponseStoreError) as in_progress:
        registry.parent_messages("resp_live")
    assert in_progress.value.code == "response_in_progress"

    registry.commit(
        "resp_live",
        _envelope("resp_live"),
        materialized_messages=[],
    )
    assert registry.delete("resp_live")["deleted"] is True
    assert registry.delete("resp_live")["deleted"] is True
    with pytest.raises(ResponseStoreError) as deleted:
        registry.get("resp_live")
    assert deleted.value.code == "response_deleted"

    for response_id in ("resp_old", "resp_new"):
        registry.begin(
            response_id,
            store=True,
            materialized_messages=[],
            cancel=lambda: None,
        )
        registry.commit(
            response_id,
            _envelope(response_id),
            materialized_messages=[],
        )
    with pytest.raises(ResponseStoreError) as evicted:
        registry.get("resp_old")
    assert evicted.value.code == "response_evicted"

    with pytest.raises(ResponseStoreError) as missing:
        registry.get("resp_never")
    assert missing.value.code == "response_not_found"


def test_registry_ttl_and_byte_pressure_are_bounded():
    now = [100.0]
    registry = ResponseRegistry(
        max_entries=10,
        max_bytes=500,
        max_entry_bytes=400,
        idle_ttl_s=5,
        clock=lambda: now[0],
    )
    registry.begin(
        "resp_big",
        store=True,
        materialized_messages=[],
        cancel=lambda: None,
    )
    registry.commit(
        "resp_big",
        _envelope("resp_big", "x" * 1000),
        materialized_messages=[],
    )
    assert registry.stats()["entries"] == 0
    assert registry.stats()["evicted_total"] == 1

    registry.begin(
        "resp_ttl",
        store=True,
        materialized_messages=[],
        cancel=lambda: None,
    )
    registry.commit(
        "resp_ttl",
        _envelope("resp_ttl"),
        materialized_messages=[],
    )
    now[0] += 6
    with pytest.raises(ResponseStoreError) as expired:
        registry.get("resp_ttl")
    assert expired.value.code == "response_evicted"
    assert registry.stats()["expired_total"] == 1


def test_registry_bounds_in_flight_count_and_lineage_bytes():
    registry = ResponseRegistry(max_in_flight=1, max_entry_bytes=80)
    registry.begin(
        "resp_live",
        store=True,
        materialized_messages=[{"role": "user", "content": "bounded"}],
        cancel=lambda: None,
    )
    stats = registry.stats()
    assert stats["in_flight"] == 1
    assert 0 < stats["in_flight_bytes"] <= stats["max_entry_bytes"]

    with pytest.raises(ResponseStoreError) as capacity:
        registry.begin(
            "resp_second",
            store=True,
            materialized_messages=[],
            cancel=lambda: None,
        )
    assert capacity.value.code == "response_store_capacity"
    assert capacity.value.status_code == 429

    registry.commit(
        "resp_live",
        _envelope("resp_live"),
        materialized_messages=[],
    )
    assert registry.stats()["in_flight_bytes"] == 0

    with pytest.raises(ResponseStoreError) as too_large:
        registry.begin(
            "resp_large",
            store=True,
            materialized_messages=[{"role": "user", "content": "x" * 100}],
            cancel=lambda: None,
        )
    assert too_large.value.code == "response_too_large"
    assert too_large.value.status_code == 413


def test_cancel_and_delete_races_settle_once_without_resurrection():
    calls = Counter()
    registry = ResponseRegistry()
    registry.begin(
        "resp_cancel",
        store=True,
        materialized_messages=[],
        cancel=lambda: calls.update(cancel=1),
    )
    first = registry.request_cancel("resp_cancel")
    second = registry.request_cancel("resp_cancel")
    assert first is second
    assert calls["cancel"] == 1
    registry.commit(
        "resp_cancel",
        {
            **_envelope("resp_cancel"),
            "status": "cancelled",
            "error": {"code": "request_cancelled", "message": "cancelled"},
        },
        materialized_messages=[],
    )
    assert registry.wait_terminal(first, 0)["status"] == "cancelled"
    assert registry.stats()["cancel_settled_total"] == 1

    registry.begin(
        "resp_deleted_live",
        store=True,
        materialized_messages=[],
        cancel=lambda: calls.update(delete_cancel=1),
    )
    registry.delete("resp_deleted_live")
    registry.commit(
        "resp_deleted_live",
        _envelope("resp_deleted_live"),
        materialized_messages=[],
    )
    with pytest.raises(ResponseStoreError) as deleted:
        registry.get("resp_deleted_live")
    assert deleted.value.code == "response_deleted"
    assert calls["delete_cancel"] == 1


def test_api_store_roundtrip_chain_and_health(monkeypatch):
    state = _fake_state()
    captured_messages: list[list[dict]] = []

    def encode(_tokenizer, messages, **_kwargs):
        captured_messages.append(
            [
                message.model_dump(exclude_none=True)
                if hasattr(message, "model_dump")
                else dict(message)
                for message in messages
            ]
        )
        return [1, 2, 3]

    generations = iter(
        [_fake_generation("First answer"), _fake_generation("Second answer")]
    )
    monkeypatch.setattr(openai, "_encode_messages", encode)
    monkeypatch.setattr(openai, "_run_generation", lambda *_a, **_kw: next(generations))
    client = TestClient(create_app(state))

    first = client.post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "input": "First question",
            "instructions": "Parent instruction",
            "store": True,
        },
    )
    assert first.status_code == 200, first.text
    first_payload = first.json()
    assert first_payload["store"] is True
    assert client.get(f"/v1/responses/{first_payload['id']}").json() == first_payload

    second = client.post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "input": "Second question",
            "instructions": "Child instruction",
            "previous_response_id": first_payload["id"],
            "store": True,
        },
    )
    assert second.status_code == 200, second.text
    second_payload = second.json()
    assert second_payload["previous_response_id"] == first_payload["id"]
    assert [message["content"] for message in captured_messages[-1]] == [
        "Child instruction",
        "First question",
        "First answer",
        "Second question",
    ]
    assert all(
        message["content"] != "Parent instruction"
        for message in captured_messages[-1]
    )

    health = client.get("/health").json()["response_store"]
    metrics = client.get("/metrics").json()["response_store"]
    assert health["entries"] == metrics["entries"] == 2
    assert health["in_flight"] == 0

    deleted = client.delete(f"/v1/responses/{first_payload['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert client.get(f"/v1/responses/{first_payload['id']}").status_code == 410
    assert client.delete(f"/v1/responses/{first_payload['id']}").status_code == 200


def _run_streaming_cancel_test(monkeypatch, *, with_client_hint: bool):
    """Shared cancel proof: exactly one cancel transition through turn.cancel_driver()."""
    state = _fake_streaming_session_state()
    state.response_registry = ResponseRegistry()
    driver_started = threading.Event()
    cancel_transitions = Counter()

    def generate(_state, _prompt_ids, **kwargs):
        driver_started.set()
        cancel_event = kwargs["cancel_event"]
        while not cancel_event.wait(0.01):
            pass
        cancel_transitions.update(worker=1)
        raise openai._StreamCancelled("cancelled_by_response_endpoint")

    monkeypatch.setattr(openai, "_encode_messages", lambda *_a, **_kw: [1, 2, 3])
    monkeypatch.setattr(openai, "_run_generation", generate)
    monkeypatch.setattr(
        state.response_registry,
        "allocate_id",
        lambda _preferred=None: "resp-cancel-test",
    )
    client = TestClient(create_app(state))
    stream_result: dict[str, object] = {}
    headers: dict[str, str] = {}
    if with_client_hint:
        headers["x-mtplx-request-id"] = "cancel-test"

    def consume_stream() -> None:
        response = client.post(
            "/v1/responses",
            headers=headers,
            json={"input": "wait", "stream": True, "store": True},
        )
        stream_result["status"] = response.status_code
        stream_result["text"] = response.text

    dashboard_before = state.dashboard.in_flight.count()

    thread = threading.Thread(target=consume_stream)
    thread.start()
    assert driver_started.wait(1.0)
    started = time.monotonic()
    cancelled = client.post("/v1/responses/resp-cancel-test/cancel")
    elapsed = time.monotonic() - started
    thread.join(timeout=2.0)
    state.generation_executor.shutdown(wait=True)

    assert elapsed < 1.0
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["error"]["code"] == "request_cancelled"
    assert stream_result["status"] == 200
    assert '"type": "response.failed"' in str(stream_result["text"])

    stats = state.response_registry.stats()
    assert stats["in_flight"] == 0
    assert stats["cancel_requests_total"] == 1
    assert stats["cancel_settled_total"] == 1
    assert cancel_transitions["worker"] == 1

    assert state.dashboard.in_flight.count() == dashboard_before

    stored = client.get("/v1/responses/resp-cancel-test").json()
    assert stored["status"] == "cancelled"

    return state, client


def test_cancel_endpoint_reaches_live_generation_turn(monkeypatch):
    _run_streaming_cancel_test(monkeypatch, with_client_hint=True)


def test_cancel_without_client_hint_uses_turn_cancel_driver(monkeypatch):
    _run_streaming_cancel_test(monkeypatch, with_client_hint=False)


def test_cancel_uses_actual_turn_handle_not_phantom_dashboard_id(monkeypatch):
    """Prove the cancel path goes through turn.cancel_driver(), not
    dashboard.in_flight.cancel() with a regenerated chat ID."""
    state = _fake_streaming_session_state()
    state.response_registry = ResponseRegistry()
    driver_started = threading.Event()
    dashboard_cancel_calls: list[str] = []
    original_dashboard_cancel = state.dashboard.in_flight.cancel

    def tracking_dashboard_cancel(request_id: str) -> bool:
        dashboard_cancel_calls.append(request_id)
        return original_dashboard_cancel(request_id)

    state.dashboard.in_flight.cancel = tracking_dashboard_cancel

    def generate(_state, _prompt_ids, **kwargs):
        driver_started.set()
        cancel_event = kwargs["cancel_event"]
        while not cancel_event.wait(0.01):
            pass
        raise openai._StreamCancelled("cancelled")

    monkeypatch.setattr(openai, "_encode_messages", lambda *_a, **_kw: [1, 2, 3])
    monkeypatch.setattr(openai, "_run_generation", generate)
    monkeypatch.setattr(
        state.response_registry,
        "allocate_id",
        lambda _preferred=None: "resp-phantom-test",
    )
    client = TestClient(create_app(state))
    stream_result: dict[str, object] = {}

    def consume_stream() -> None:
        response = client.post(
            "/v1/responses",
            json={"input": "wait", "stream": True, "store": True},
        )
        stream_result["status"] = response.status_code
        stream_result["text"] = response.text

    thread = threading.Thread(target=consume_stream)
    thread.start()
    assert driver_started.wait(1.0)
    cancelled = client.post("/v1/responses/resp-phantom-test/cancel")
    thread.join(timeout=2.0)
    state.generation_executor.shutdown(wait=True)

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    phantom_calls = [
        call_id for call_id in dashboard_cancel_calls
        if call_id.startswith("chatcmpl-")
    ]
    assert phantom_calls == [], (
        f"cancel_generation() called dashboard.in_flight.cancel() "
        f"with phantom chatcmpl ID(s): {phantom_calls}"
    )
