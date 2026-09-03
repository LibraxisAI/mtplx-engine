"""Render Chat Completions results as Responses objects and SSE events."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from mtplx.server.core.events import (
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

from .schema import ResponsesRequest


def _item_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _usage(chat_usage: dict[str, Any] | None) -> dict[str, Any]:
    source = chat_usage or {}
    prompt = int(source.get("prompt_tokens") or 0)
    output = int(source.get("completion_tokens") or 0)
    prompt_details = source.get("prompt_tokens_details") or {}
    output_details = source.get("completion_tokens_details") or {}
    return {
        "input_tokens": prompt,
        "input_tokens_details": {
            "cached_tokens": int(prompt_details.get("cached_tokens") or 0)
        },
        "output_tokens": output,
        "output_tokens_details": {
            "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0)
        },
        "total_tokens": int(source.get("total_tokens") or prompt + output),
    }


def _response_error(chat_error: Any) -> dict[str, str]:
    """Normalize Chat-only failures to the Responses wire contract."""

    if isinstance(chat_error, dict):
        message = chat_error.get("message")
    else:
        message = chat_error
    return {
        "code": "server_error",
        "message": str(message or "Response generation failed"),
    }


def _reasoning_item(text: str, *, item_id: str | None = None, status: str = "completed") -> dict[str, Any]:
    return {
        "id": item_id or _item_id("rs"),
        "type": "reasoning",
        "status": status,
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    }


def _message_item(text: str, *, item_id: str | None = None, status: str = "completed") -> dict[str, Any]:
    return {
        "id": item_id or _item_id("msg"),
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }


def _function_item(call: dict[str, Any], *, item_id: str | None = None, status: str = "completed") -> dict[str, Any]:
    function = call.get("function") or {}
    return {
        "id": item_id or _item_id("fc"),
        "type": "function_call",
        "status": status,
        "call_id": str(call.get("id") or _item_id("call")),
        "name": str(function.get("name") or ""),
        "arguments": str(function.get("arguments") or ""),
    }


def _base_response(
    *,
    response_id: str,
    request: ResponsesRequest,
    model: str,
    created_at: int,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    mtplx_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "model": model,
        "output": output,
        "parallel_tool_calls": (
            True if request.parallel_tool_calls is None else request.parallel_tool_calls
        ),
        "previous_response_id": None,
        "reasoning": request.reasoning,
        "store": False,
        "temperature": request.temperature,
        "text": request.text or {"format": {"type": "text"}},
        "tool_choice": request.tool_choice or "auto",
        "tools": request.tools or [],
        "top_p": request.top_p,
        "truncation": "disabled",
        "usage": usage,
        "metadata": request.metadata or {},
    }
    if mtplx_stats is not None:
        payload["mtplx_stats"] = mtplx_stats
    return payload


def response_from_chat_completion(
    chat: dict[str, Any],
    *,
    request: ResponsesRequest,
    response_id: str,
    created_at: int,
) -> dict[str, Any]:
    """Convert a completed Chat Completions envelope without losing channels."""

    choices = chat.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    output: list[dict[str, Any]] = []
    reasoning = str(message.get("reasoning_content") or "")
    if reasoning:
        output.append(_reasoning_item(reasoning))
    text = str(message.get("content") or "")
    if text or not message.get("tool_calls"):
        output.append(_message_item(text))
    for call in message.get("tool_calls") or []:
        output.append(_function_item(call))
    incomplete = choice.get("finish_reason") == "length"
    payload = _base_response(
        response_id=response_id,
        request=request,
        model=str(chat.get("model") or request.model or ""),
        created_at=created_at,
        status="incomplete" if incomplete else "completed",
        output=output,
        usage=_usage(chat.get("usage")),
        mtplx_stats=chat.get("mtplx_stats") or {},
    )
    if incomplete:
        payload["incomplete_details"] = {"reason": "max_output_tokens"}
    return payload


async def _chat_payloads(body_iterator: AsyncIterator[Any]) -> AsyncIterator[dict[str, Any]]:
    buffer = ""
    async for raw in body_iterator:
        buffer += raw.decode() if isinstance(raw, bytes) else str(raw)
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            data = "\n".join(
                line.removeprefix("data: ")
                for line in frame.splitlines()
                if line.startswith("data: ")
            ).strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


@dataclass
class _StreamState:
    response_id: str
    request: ResponsesRequest
    model: str
    created_at: int
    sequence: int = 0
    text_id: str | None = None
    text: str = ""
    reasoning_id: str | None = None
    reasoning: str = ""
    tools: dict[int, dict[str, Any]] = field(default_factory=dict)
    output_order: list[tuple[str, int]] = field(default_factory=list)
    output_indices: dict[tuple[str, int], int] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=lambda: _usage(None))
    mtplx_stats: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None

    def event(self, event_type: str, **payload: Any) -> str:
        event = {"type": event_type, "sequence_number": self.sequence, **payload}
        self.sequence += 1
        return f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

    def register_output(self, kind: str, key: int = 0) -> int:
        identity = (kind, key)
        if identity not in self.output_indices:
            self.output_indices[identity] = len(self.output_order)
            self.output_order.append(identity)
        return self.output_indices[identity]

    def output_index(self, kind: str, key: int = 0) -> int:
        return self.output_indices[(kind, key)]

    def output(self, *, status: str = "completed") -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for kind, key in self.output_order:
            if kind == "reasoning" and self.reasoning_id is not None:
                items.append(
                    _reasoning_item(
                        self.reasoning,
                        item_id=self.reasoning_id,
                        status=status,
                    )
                )
            elif kind == "text" and self.text_id is not None:
                items.append(
                    _message_item(self.text, item_id=self.text_id, status=status)
                )
            elif kind == "tool":
                tool = self.tools[key]
                items.append(
                    _function_item(
                        {
                            "id": tool["call_id"],
                            "function": {
                                "name": tool["name"],
                                "arguments": tool["arguments"],
                            },
                        },
                        item_id=tool["id"],
                        status=status,
                    )
                )
        return items

    def response(
        self, *, status: str, output_status: str | None = None
    ) -> dict[str, Any]:
        response = _base_response(
            response_id=self.response_id,
            request=self.request,
            model=self.model,
            created_at=self.created_at,
            status=status,
            output=self.output(status=output_status or status),
            usage=self.usage,
            mtplx_stats=self.mtplx_stats,
        )
        if status == "incomplete":
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        return response


async def responses_stream_from_chat_sse(
    body_iterator: AsyncIterator[Any],
    *,
    request: ResponsesRequest,
    response_id: str,
    model: str,
    created_at: int | None = None,
) -> AsyncIterator[str]:
    """Translate the existing temporal Chat SSE stream in-process."""

    state = _StreamState(
        response_id=response_id,
        request=request,
        model=model,
        created_at=int(created_at or time.time()),
    )
    yield state.event("response.created", response=state.response(status="in_progress"))
    yield state.event("response.in_progress", response=state.response(status="in_progress"))
    try:
        async for chat in _chat_payloads(body_iterator):
            if chat.get("error"):
                failed = state.response(status="failed", output_status="incomplete")
                failed["error"] = _response_error(chat["error"])
                yield state.event("response.failed", response=failed)
                return
            if isinstance(chat.get("mtplx_progress"), dict):
                # Preserve Chat's liveness signal as an SSE comment. SDKs
                # ignore comments, while proxies keep a long burst alive.
                yield ": mtplx-heartbeat\n\n"
                continue
            if chat.get("model"):
                state.model = str(chat["model"])
            choices = chat.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            reasoning_delta = str(delta.get("reasoning_content") or "")
            if reasoning_delta:
                if state.reasoning_id is None:
                    state.reasoning_id = _item_id("rs")
                    output_index = state.register_output("reasoning")
                    item = _reasoning_item("", item_id=state.reasoning_id, status="in_progress")
                    yield state.event(
                        "response.output_item.added",
                        output_index=output_index,
                        item=item,
                    )
                output_index = state.output_index("reasoning")
                state.reasoning += reasoning_delta
                yield state.event(
                    "response.reasoning_text.delta",
                    item_id=state.reasoning_id,
                    output_index=output_index,
                    content_index=0,
                    delta=reasoning_delta,
                )
            text_delta = str(delta.get("content") or "")
            if text_delta:
                if state.text_id is None:
                    state.text_id = _item_id("msg")
                    output_index = state.register_output("text")
                    item = _message_item("", item_id=state.text_id, status="in_progress")
                    yield state.event(
                        "response.output_item.added",
                        output_index=output_index,
                        item=item,
                    )
                    yield state.event(
                        "response.content_part.added",
                        item_id=state.text_id,
                        output_index=output_index,
                        content_index=0,
                        part=item["content"][0],
                    )
                output_index = state.output_index("text")
                state.text += text_delta
                yield state.event(
                    "response.output_text.delta",
                    item_id=state.text_id,
                    output_index=output_index,
                    content_index=0,
                    delta=text_delta,
                    logprobs=[],
                )
            for call_delta in delta.get("tool_calls") or []:
                index = int(call_delta.get("index") or 0)
                function = call_delta.get("function") or {}
                tool = state.tools.get(index)
                if tool is None:
                    tool = {
                        "id": _item_id("fc"),
                        "call_id": str(call_delta.get("id") or _item_id("call")),
                        "name": str(function.get("name") or ""),
                        "arguments": "",
                    }
                    state.tools[index] = tool
                    output_index = state.register_output("tool", index)
                    yield state.event(
                        "response.output_item.added",
                        output_index=output_index,
                        item=_function_item(
                            {
                                "id": tool["call_id"],
                                "function": {"name": tool["name"], "arguments": ""},
                            },
                            item_id=tool["id"],
                            status="in_progress",
                        ),
                    )
                if function.get("name"):
                    tool["name"] = str(function["name"])
                output_index = state.output_index("tool", index)
                arguments_delta = str(function.get("arguments") or "")
                if arguments_delta:
                    tool["arguments"] += arguments_delta
                    yield state.event(
                        "response.function_call_arguments.delta",
                        item_id=tool["id"],
                        output_index=output_index,
                        delta=arguments_delta,
                    )
            if chat.get("usage"):
                state.usage = _usage(chat["usage"])
            if chat.get("mtplx_stats"):
                state.mtplx_stats = chat["mtplx_stats"]
            if choice.get("finish_reason"):
                state.finish_reason = str(choice["finish_reason"])
    finally:
        if hasattr(body_iterator, "aclose"):
            await body_iterator.aclose()

    for output_index, (kind, key) in enumerate(state.output_order):
        if kind == "reasoning" and state.reasoning_id is not None:
            yield state.event(
                "response.reasoning_text.done",
                item_id=state.reasoning_id,
                output_index=output_index,
                content_index=0,
                text=state.reasoning,
            )
            yield state.event(
                "response.output_item.done",
                output_index=output_index,
                item=_reasoning_item(state.reasoning, item_id=state.reasoning_id),
            )
        elif kind == "text" and state.text_id is not None:
            item = _message_item(state.text, item_id=state.text_id)
            yield state.event(
                "response.output_text.done",
                item_id=state.text_id,
                output_index=output_index,
                content_index=0,
                text=state.text,
                logprobs=[],
            )
            yield state.event(
                "response.content_part.done",
                item_id=state.text_id,
                output_index=output_index,
                content_index=0,
                part=item["content"][0],
            )
            yield state.event(
                "response.output_item.done",
                output_index=output_index,
                item=item,
            )
        elif kind == "tool":
            tool = state.tools[key]
            item = _function_item(
                {
                    "id": tool["call_id"],
                    "function": {
                        "name": tool["name"],
                        "arguments": tool["arguments"],
                    },
                },
                item_id=tool["id"],
            )
            yield state.event(
                "response.function_call_arguments.done",
                item_id=tool["id"],
                output_index=output_index,
                name=tool["name"],
                arguments=tool["arguments"],
            )
            yield state.event(
                "response.output_item.done",
                output_index=output_index,
                item=item,
            )
    terminal_status = "incomplete" if state.finish_reason == "length" else "completed"
    yield state.event(
        "response.incomplete" if terminal_status == "incomplete" else "response.completed",
        response=state.response(status=terminal_status),
    )


async def responses_stream_from_turn_events(
    events: AsyncIterator[TurnEvent],
    *,
    request: ResponsesRequest,
    response_id: str,
    model: str,
    created_at: int | None = None,
) -> AsyncIterator[str]:
    """Render Responses SSE directly from protocol-neutral TurnEvents."""

    state = _StreamState(
        response_id=response_id,
        request=request,
        model=model,
        created_at=int(created_at or time.time()),
    )
    yield state.event("response.created", response=state.response(status="in_progress"))
    yield state.event("response.in_progress", response=state.response(status="in_progress"))

    async for ev in events:
        if isinstance(ev, TurnStarted):
            state.model = ev.model
            continue

        if isinstance(ev, ReasoningDelta):
            if state.reasoning_id is None:
                state.reasoning_id = _item_id("rs")
                output_index = state.register_output("reasoning")
                item = _reasoning_item("", item_id=state.reasoning_id, status="in_progress")
                yield state.event(
                    "response.output_item.added",
                    output_index=output_index,
                    item=item,
                )
            output_index = state.output_index("reasoning")
            state.reasoning += ev.delta
            yield state.event(
                "response.reasoning_text.delta",
                item_id=state.reasoning_id,
                output_index=output_index,
                content_index=0,
                delta=ev.delta,
            )
            continue

        if isinstance(ev, TextDelta):
            if state.text_id is None:
                state.text_id = _item_id("msg")
                output_index = state.register_output("text")
                item = _message_item("", item_id=state.text_id, status="in_progress")
                yield state.event(
                    "response.output_item.added",
                    output_index=output_index,
                    item=item,
                )
                yield state.event(
                    "response.content_part.added",
                    item_id=state.text_id,
                    output_index=output_index,
                    content_index=0,
                    part=item["content"][0],
                )
            output_index = state.output_index("text")
            state.text += ev.delta
            yield state.event(
                "response.output_text.delta",
                item_id=state.text_id,
                output_index=output_index,
                content_index=0,
                delta=ev.delta,
                logprobs=[],
            )
            continue

        if isinstance(ev, ToolCallDelta):
            tool = state.tools.get(ev.index)
            if tool is None:
                tool = {
                    "id": _item_id("fc"),
                    "call_id": str(ev.call_id or _item_id("call")),
                    "name": str(ev.name or ""),
                    "arguments": "",
                }
                state.tools[ev.index] = tool
                output_index = state.register_output("tool", ev.index)
                yield state.event(
                    "response.output_item.added",
                    output_index=output_index,
                    item=_function_item(
                        {
                            "id": tool["call_id"],
                            "function": {"name": tool["name"], "arguments": ""},
                        },
                        item_id=tool["id"],
                        status="in_progress",
                    ),
                )
            if ev.name:
                tool["name"] = str(ev.name)
            output_index = state.output_index("tool", ev.index)
            if ev.arguments_delta:
                tool["arguments"] += ev.arguments_delta
                yield state.event(
                    "response.function_call_arguments.delta",
                    item_id=tool["id"],
                    output_index=output_index,
                    delta=ev.arguments_delta,
                )
            continue

        if isinstance(ev, UsageUpdate):
            state.usage = _usage({
                "prompt_tokens": ev.prompt_tokens,
                "completion_tokens": ev.completion_tokens,
            })
            continue

        if isinstance(ev, TurnCompleted):
            state.finish_reason = ev.finish_reason
            if ev.usage is not None:
                state.usage = _usage({
                    "prompt_tokens": ev.usage.prompt_tokens,
                    "completion_tokens": ev.usage.completion_tokens,
                })
            break

        if isinstance(ev, TurnFailed):
            failed = state.response(status="failed", output_status="incomplete")
            failed["error"] = {"code": ev.code or "server_error", "message": ev.error}
            yield state.event("response.failed", response=failed)
            return

        if isinstance(ev, TurnCancelled):
            failed = state.response(status="failed", output_status="incomplete")
            failed["error"] = {"code": "request_cancelled", "message": ev.reason}
            yield state.event("response.failed", response=failed)
            return

    for output_index, (kind, key) in enumerate(state.output_order):
        if kind == "reasoning" and state.reasoning_id is not None:
            yield state.event(
                "response.reasoning_text.done",
                item_id=state.reasoning_id,
                output_index=output_index,
                content_index=0,
                text=state.reasoning,
            )
            yield state.event(
                "response.output_item.done",
                output_index=output_index,
                item=_reasoning_item(state.reasoning, item_id=state.reasoning_id),
            )
        elif kind == "text" and state.text_id is not None:
            item = _message_item(state.text, item_id=state.text_id)
            yield state.event(
                "response.output_text.done",
                item_id=state.text_id,
                output_index=output_index,
                content_index=0,
                text=state.text,
                logprobs=[],
            )
            yield state.event(
                "response.content_part.done",
                item_id=state.text_id,
                output_index=output_index,
                content_index=0,
                part=item["content"][0],
            )
            yield state.event(
                "response.output_item.done",
                output_index=output_index,
                item=item,
            )
        elif kind == "tool":
            tool = state.tools[key]
            item = _function_item(
                {
                    "id": tool["call_id"],
                    "function": {
                        "name": tool["name"],
                        "arguments": tool["arguments"],
                    },
                },
                item_id=tool["id"],
            )
            yield state.event(
                "response.function_call_arguments.done",
                item_id=tool["id"],
                output_index=output_index,
                name=tool["name"],
                arguments=tool["arguments"],
            )
            yield state.event(
                "response.output_item.done",
                output_index=output_index,
                item=item,
            )
    terminal_status = "incomplete" if state.finish_reason == "length" else "completed"
    yield state.event(
        "response.incomplete" if terminal_status == "incomplete" else "response.completed",
        response=state.response(status=terminal_status),
    )
