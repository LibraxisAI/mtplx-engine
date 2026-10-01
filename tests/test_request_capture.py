"""Tests for privacy-safe per-request capture."""

from __future__ import annotations

import json
import os

from mtplx import request_capture
from mtplx.request_capture import CaptureOptions

_CONTENT_ENV = (
    "MTPLX_REQUEST_CAPTURE_INCLUDE_PROMPT_TOKENS",
    "MTPLX_REQUEST_CAPTURE_INCLUDE_COMPLETION_TOKENS",
    "MTPLX_REQUEST_CAPTURE_INCLUDE_PROMPT_TEXT",
    "MTPLX_REQUEST_CAPTURE_INCLUDE_RESPONSE_TEXT",
    "MTPLX_REQUEST_CAPTURE_INCLUDE_MESSAGES",
    "MTPLX_REQUEST_CAPTURE_INCLUDE_EXCEPTION_TEXT",
    "MTPLX_REQUEST_CAPTURE_PROMPT_TOKEN_LIMIT",
    "MTPLX_REQUEST_CAPTURE_COMPLETION_TOKEN_LIMIT",
)


def _enable(monkeypatch, tmp_path, keep="200"):
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_KEEP", keep)
    for name in _CONTENT_ENV:
        monkeypatch.delenv(name, raising=False)
    request_capture._PATHS_BY_ID.clear()


def _record(tmp_path):
    files = [path for path in tmp_path.iterdir() if path.suffix == ".json"]
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_capture_defaults_to_content_off(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request(
        "chatcmpl-abc",
        {
            "prompt_token_ids": [1, 2, 3],
            "prompt_text": "private prompt",
            "messages": [{"role": "user", "content": "private message"}],
            "max_tokens": 64,
        },
    )

    record = _record(tmp_path)
    assert record["capture_version"] == 2
    assert record["phase"] == "dispatched"
    assert record["prompt_token_count"] == 3
    assert record["prompt_tokens_sha256"] == request_capture.stable_json_digest(
        [1, 2, 3]
    )
    assert "prompt_token_ids" not in record
    assert "prompt_text" not in record
    assert "prompt_text_excerpt" not in record
    assert "messages" not in record
    assert record["capture_content_policy"]["prompt_tokens"] is False
    assert record["max_tokens"] == 64
    assert "private" not in json.dumps(record)


def test_completion_helper_is_privacy_safe_by_default(monkeypatch):
    for name in _CONTENT_ENV:
        monkeypatch.delenv(name, raising=False)

    result = request_capture.completion_token_ids([8, 13, 21])

    assert result == {
        "completion_token_count": 3,
        "completion_tokens_sha256": request_capture.stable_json_digest([8, 13, 21]),
    }


def test_completion_helper_retains_one_argument_and_never_raises(monkeypatch):
    for name in _CONTENT_ENV:
        monkeypatch.delenv(name, raising=False)

    assert request_capture.completion_token_ids(None)["completion_token_count"] == 0
    assert request_capture.completion_token_ids(7)["completion_token_count"] == 0
    assert request_capture.completion_token_ids([1, "bad"])[
        "completion_token_count"
    ] == 0

    class BrokenTokens:
        def __iter__(self):
            raise RuntimeError("broken iterator")

    assert request_capture.completion_token_ids(BrokenTokens())[
        "completion_token_count"
    ] == 0


def test_completion_tokens_require_their_own_opt_in(monkeypatch):
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_INCLUDE_PROMPT_TOKENS", "1")
    monkeypatch.delenv(
        "MTPLX_REQUEST_CAPTURE_INCLUDE_COMPLETION_TOKENS", raising=False
    )
    assert "completion_token_ids" not in request_capture.completion_token_ids([1, 2])

    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_INCLUDE_COMPLETION_TOKENS", "true")
    result = request_capture.completion_token_ids([1, 2])
    assert result["completion_token_ids"] == [1, 2]
    assert result["completion_tokens_clipped"] is False


def test_token_opt_ins_are_independent_and_bounded(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    options = CaptureOptions(
        include_prompt_tokens=True,
        include_completion_tokens=True,
        prompt_token_limit=2,
        completion_token_limit=1,
    )
    request_capture.capture_request(
        "chatcmpl-optin",
        {"prompt_token_ids": [1, 2, 3]},
        options=options,
    )
    request_capture.capture_outcome(
        "chatcmpl-optin",
        {"completion_token_ids": [4, 5, 6]},
        options=options,
    )

    record = _record(tmp_path)
    assert record["prompt_token_ids"] == [1, 2]
    assert record["prompt_tokens_clipped"] is True
    assert record["outcome"]["completion_token_ids"] == [4]
    assert record["outcome"]["completion_tokens_clipped"] is True


def test_direct_completion_token_payload_is_sanitized(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request("chatcmpl-direct", {"max_tokens": 8})
    request_capture.capture_outcome(
        "chatcmpl-direct",
        {"completion_token_ids": [3, 1, 4], "completion_token_count": 999},
    )

    outcome = _record(tmp_path)["outcome"]
    assert "completion_token_ids" not in outcome
    assert outcome["completion_token_count"] == 3
    assert outcome["completion_tokens_sha256"] == (
        request_capture.stable_json_digest([3, 1, 4])
    )


def test_nested_token_ids_are_sanitized(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request(
        "chatcmpl-nested",
        {
            "metadata": {
                "prompt_token_ids": [10, 20],
                "child": {"completion_token_ids": [30]},
            }
        },
    )

    metadata = _record(tmp_path)["metadata"]
    assert "prompt_token_ids" not in metadata
    assert metadata["prompt_token_count"] == 2
    assert "completion_token_ids" not in metadata["child"]
    assert metadata["child"]["completion_token_count"] == 1


def test_capture_then_outcome_redacts_response_and_exception(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request("chatcmpl-abc", {"max_tokens": 64})
    request_capture.capture_outcome(
        "chatcmpl-abc",
        {
            "finish_reason": "stop",
            "completion_tokens": 5,
            "text": "private response",
            "exception_message": "token=secret",
        },
    )

    record = _record(tmp_path)
    outcome = record["outcome"]
    assert record["phase"] == "completed"
    assert outcome["finish_reason"] == "stop"
    assert "text" not in outcome
    assert "exception_message" not in outcome
    assert "text_sha256" in outcome
    assert "exception_message_sha256" in outcome
    assert "private response" not in json.dumps(record)
    assert "token=secret" not in json.dumps(record)


def test_explicit_text_options_are_bounded(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request(
        "chatcmpl-text-optin",
        {"prompt_text": "abcdefghij"},
        options=CaptureOptions(
            include_prompt_text=True,
            text_head_chars=3,
            text_tail_chars=2,
        ),
    )

    assert _record(tmp_path)["prompt_text_excerpt"] == {
        "text_head": "abc",
        "text_tail": "ij",
        "text_chars": 10,
        "text_clipped": True,
    }


def test_large_clipped_outcome_cannot_bypass_content_off(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request("chatcmpl-large", {"max_tokens": 64})
    clipped = request_capture.clip_text_head_tail("private-response-" * 500)
    request_capture.capture_outcome("chatcmpl-large", clipped)

    outcome = _record(tmp_path)["outcome"]
    assert "text_head" not in outcome
    assert "text_tail" not in outcome
    assert "text_head_sha256" in outcome
    assert "text_tail_sha256" in outcome


def test_nested_credentials_are_always_redacted(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request(
        "chatcmpl-redact",
        {
            "observability": {"authorization": "Bearer secret", "mode": "mtp"},
            "messages": [
                {
                    "role": "user",
                    "content": "hello",
                    "api_key": "message-secret",
                    "completion_token_ids": [7, 8],
                }
            ],
        },
        options=CaptureOptions(include_messages=True),
    )

    record = _record(tmp_path)
    assert record["observability"]["authorization"] == "<redacted>"
    assert record["messages"][0]["api_key"] == "<redacted>"
    assert "completion_token_ids" not in record["messages"][0]
    assert record["messages"][0]["completion_token_count"] == 2
    assert "Bearer secret" not in json.dumps(record)
    assert "message-secret" not in json.dumps(record)


def test_dispatch_record_survives_missing_outcome(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request("chatcmpl-hang", {"prompt_token_ids": [7]})

    record = _record(tmp_path)
    assert record["phase"] == "dispatched"
    assert "outcome" not in record


def test_ring_prunes_to_keep_without_deleting(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path, keep="3")
    for index in range(6):
        request_capture.capture_request(f"r{index}", {"i": index})

    live = [name for name in os.listdir(tmp_path) if name.endswith(".json")]
    assert len(live) == 3
    assert len(os.listdir(tmp_path / "pruned")) == 3
    assert len(request_capture._PATHS_BY_ID) == 3


def test_completed_capture_releases_request_path(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request("chatcmpl-complete", {"a": 1})
    assert "chatcmpl-complete" in request_capture._PATHS_BY_ID

    request_capture.capture_outcome("chatcmpl-complete", {"b": 2})
    assert "chatcmpl-complete" not in request_capture._PATHS_BY_ID


def test_disabled_is_a_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("MTPLX_REQUEST_CAPTURE_DIR", raising=False)
    request_capture.capture_request("x", {"a": 1})
    request_capture.capture_outcome("x", {"b": 2})
    assert list(tmp_path.iterdir()) == []


def test_stable_json_digest_ignores_mapping_order():
    assert request_capture.stable_json_digest({"b": 2, "a": 1}) == (
        request_capture.stable_json_digest({"a": 1, "b": 2})
    )


def test_clip_text_head_tail():
    small = request_capture.clip_text_head_tail("hello", head=10, tail=10)
    assert small == {"text": "hello", "text_clipped": False}
    big = request_capture.clip_text_head_tail("a" * 100, head=10, tail=10)
    assert big["text_clipped"] and big["text_chars"] == 100
    assert len(big["text_head"]) == 10 and len(big["text_tail"]) == 10


def test_unsafe_request_ids_are_sanitized(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request("../../etc/passwd", {"a": 1})
    files = [name for name in os.listdir(tmp_path) if name.endswith(".json")]
    assert len(files) == 1
    assert ".." not in files[0] and "/" not in files[0]


def test_registry_forgets_pruned_requests(monkeypatch, tmp_path):
    """The path registry must not grow past the ring: pruned ids are dropped."""
    _enable(monkeypatch, tmp_path, keep="3")
    for i in range(7):
        request_capture.capture_request(f"chatcmpl-{i}", {"prompt_token_ids": [i]})
    live = [f for f in os.listdir(tmp_path) if f.endswith(".json")]
    assert len(live) == 3
    assert len(request_capture._PATHS_BY_ID) == 3
    assert set(request_capture._PATHS_BY_ID) == {"chatcmpl-4", "chatcmpl-5", "chatcmpl-6"}

    # An outcome for a pruned id is a no-op and leaves the pruned envelope intact.
    request_capture.capture_outcome("chatcmpl-0", {"finish_reason": "stop"})
    pruned = sorted(os.listdir(tmp_path / "pruned"))
    assert len(pruned) == 4
    rec = json.load(open(tmp_path / "pruned" / pruned[0]))
    assert rec["phase"] == "dispatched"
    assert "chatcmpl-0" not in request_capture._PATHS_BY_ID


def _server_shaped_outcome(completion):
    """The outcome dict exactly as mtplx/server/openai.py assembles it."""
    return {
        "scheduler_lane": "serial",
        **request_capture.completion_token_ids(completion),
        "completion_tokens": len(completion),
        "finish_reason": "stop",
        **request_capture.clip_text_head_tail("private answer"),
    }


def test_opted_in_token_ids_are_whole_for_replay(monkeypatch, tmp_path):
    """A replay needs every id. An agent prompt is far past 2,048 tokens, so
    an opt-in with no limit set keeps both sequences whole."""
    _enable(monkeypatch, tmp_path)
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_INCLUDE_PROMPT_TOKENS", "1")
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_INCLUDE_COMPLETION_TOKENS", "1")
    prompt = list(range(5000))
    completion = list(range(3000))

    request_capture.capture_request(
        "chatcmpl-replay", {"prompt_len": len(prompt), "prompt_token_ids": prompt}
    )
    request_capture.capture_outcome(
        "chatcmpl-replay", _server_shaped_outcome(completion)
    )

    record = _record(tmp_path)
    assert record["prompt_token_ids"] == prompt
    assert record["prompt_tokens_clipped"] is False
    outcome = record["outcome"]
    assert outcome["completion_token_ids"] == completion
    assert outcome["completion_tokens_clipped"] is False
    assert outcome["completion_token_count"] == 3000
    assert "private answer" not in json.dumps(record)


def test_env_limits_clip_once_and_keep_the_true_count(monkeypatch, tmp_path):
    """The helper's result passes through the outcome sanitizer. The count,
    the digest and the clipped flag must still describe the full sequence."""
    _enable(monkeypatch, tmp_path)
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_INCLUDE_PROMPT_TOKENS", "1")
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_INCLUDE_COMPLETION_TOKENS", "1")
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_PROMPT_TOKEN_LIMIT", "4")
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_COMPLETION_TOKEN_LIMIT", "2")
    prompt = list(range(10))
    completion = [7, 8, 9, 10, 11]

    request_capture.capture_request("chatcmpl-limit", {"prompt_token_ids": prompt})
    request_capture.capture_outcome(
        "chatcmpl-limit", _server_shaped_outcome(completion)
    )

    record = _record(tmp_path)
    assert record["prompt_token_ids"] == [0, 1, 2, 3]
    assert record["prompt_tokens_clipped"] is True
    assert record["prompt_token_count"] == 10
    outcome = record["outcome"]
    assert outcome["completion_token_ids"] == [7, 8]
    assert outcome["completion_tokens_clipped"] is True
    assert outcome["completion_token_count"] == 5
    assert outcome["completion_tokens_sha256"] == (
        request_capture.stable_json_digest(completion)
    )


def test_unreadable_limit_means_no_limit(monkeypatch):
    for name in _CONTENT_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_PROMPT_TOKEN_LIMIT", "many")
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_COMPLETION_TOKEN_LIMIT", "-3")

    options = CaptureOptions.from_env()

    assert options.prompt_token_limit is None
    assert options.completion_token_limit == 0


def test_server_shaped_outcome_is_content_free_by_default(monkeypatch, tmp_path):
    """The three server call sites merge the helper and the text clip into one
    outcome. With no opt-in, nothing from the answer reaches the file."""
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request(
        "chatcmpl-default", {"prompt_len": 3, "prompt_token_ids": [11, 12, 13]}
    )
    request_capture.capture_outcome(
        "chatcmpl-default", _server_shaped_outcome([21, 22])
    )

    record = _record(tmp_path)
    text = json.dumps(record)
    assert "prompt_token_ids" not in text
    assert "completion_token_ids" not in text
    assert "private answer" not in text
    assert record["prompt_len"] == 3
    assert record["prompt_token_count"] == 3
    assert record["outcome"]["completion_token_count"] == 2
    assert record["outcome"]["completion_tokens"] == 2
    assert record["outcome"]["finish_reason"] == "stop"
    assert record["outcome"]["text_chars"] == len("private answer")

