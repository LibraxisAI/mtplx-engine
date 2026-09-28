"""Translate Responses requests into the existing Chat Completions turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mtplx.reasoning_effort import normalize_reasoning_effort

from .schema import ResponsesRequest


@dataclass(frozen=True)
class ResponsesProtocolError(ValueError):
    message: str
    param: str | None = None
    code: str = "unsupported_parameter"

    def __str__(self) -> str:
        return self.message

    def payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "param": self.param,
                "code": self.code,
            }
        }


_UNSUPPORTED_TOP_LEVEL = {
    "background",
    "conversation",
    "include",
    "max_completion_tokens",
    "max_tokens",
    "modalities",
    "prompt",
    "provider",
    "service_tier",
    "system_instruction",
    "truncation",
}

# Effort strings that mean "thinking disabled", never an effort tier.
_REASONING_DISABLED_EFFORTS = {"off", "none"}


def _reject(param: str, message: str, *, code: str = "unsupported_parameter") -> None:
    raise ResponsesProtocolError(message=message, param=param, code=code)


def _text_part(part: Mapping[str, Any], *, param: str) -> str:
    value = part.get("text")
    if not isinstance(value, str):
        _reject(param, f"{param}.text must be a string", code="invalid_type")
    return value


def _content_parts(content: Any, *, param: str) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        _reject(param, f"{param} must be a string or an array", code="invalid_type")

    translated: list[dict[str, Any]] = []
    for index, raw_part in enumerate(content):
        part_param = f"{param}[{index}]"
        if isinstance(raw_part, str):
            translated.append({"type": "text", "text": raw_part})
            continue
        if not isinstance(raw_part, Mapping):
            _reject(part_param, f"{part_param} must be an object", code="invalid_type")
        part_type = str(raw_part.get("type") or "")
        if part_type in {"input_text", "output_text", "text"}:
            translated.append(
                {"type": "text", "text": _text_part(raw_part, param=part_param)}
            )
            continue
        if part_type in {"input_image", "image_url"}:
            if raw_part.get("file_id") is not None:
                _reject(
                    f"{part_param}.file_id",
                    "file_id image inputs are not supported; use an http(s) or data URL",
                )
            image_url = raw_part.get("image_url")
            if isinstance(image_url, Mapping):
                image_url = image_url.get("url")
            if not isinstance(image_url, str) or not image_url:
                _reject(
                    f"{part_param}.image_url",
                    "input_image requires a non-empty image_url",
                    code="invalid_type",
                )
            translated.append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )
            continue
        if part_type in {"input_file", "file"} or "file_id" in raw_part:
            _reject(part_param, "file and file_id inputs are not supported")
        _reject(part_param, f"unsupported Responses content part type {part_type!r}")
    return translated


def _message(item: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    role = str(item.get("role") or "user")
    if role == "developer":
        role = "system"
    if role not in {"system", "user", "assistant", "tool"}:
        _reject(f"input[{index}].role", f"unsupported message role {role!r}")
    return {
        "role": role,
        "content": _content_parts(item.get("content", ""), param=f"input[{index}].content"),
    }


def _function_call(item: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    name = item.get("name")
    arguments = item.get("arguments", "")
    if not isinstance(name, str) or not name:
        _reject(f"input[{index}].name", "function_call.name must be a non-empty string")
    if not isinstance(arguments, str):
        _reject(
            f"input[{index}].arguments",
            "function_call.arguments must be a JSON string",
            code="invalid_type",
        )
    call_id = str(item.get("call_id") or item.get("id") or f"call_input_{index}")
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _function_call_output(item: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        _reject(
            f"input[{index}].call_id",
            "function_call_output.call_id must be a non-empty string",
        )
    output = item.get("output", "")
    if not isinstance(output, str):
        _reject(
            f"input[{index}].output",
            "function_call_output.output must be a string",
            code="invalid_type",
        )
    return {"role": "tool", "tool_call_id": call_id, "content": output}


def _messages(request: ResponsesRequest) -> list[dict[str, Any]]:
    raw_input = request.input
    if raw_input is None:
        _reject("input", "input must not be empty", code="invalid_value")
    if isinstance(raw_input, str):
        messages = [{"role": "user", "content": raw_input}]
    elif isinstance(raw_input, list):
        if raw_input and all(
            isinstance(item, Mapping)
            and str(item.get("type") or "")
            in {"input_text", "input_image", "image_url", "text"}
            for item in raw_input
        ):
            messages = [
                {
                    "role": "user",
                    "content": _content_parts(raw_input, param="input"),
                }
            ]
        else:
            messages = []
            pending_tool_calls: list[dict[str, Any]] = []

            def flush_tool_calls() -> None:
                if not pending_tool_calls:
                    return
                messages.append(
                    {"role": "assistant", "content": "", "tool_calls": list(pending_tool_calls)}
                )
                pending_tool_calls.clear()

            for index, raw_item in enumerate(raw_input):
                if not isinstance(raw_item, Mapping):
                    _reject(
                        f"input[{index}]",
                        f"input[{index}] must be an object",
                        code="invalid_type",
                    )
                item_type = str(raw_item.get("type") or "message")
                if item_type == "message":
                    flush_tool_calls()
                    messages.append(_message(raw_item, index=index))
                elif item_type == "function_call":
                    pending_tool_calls.extend(_function_call(raw_item, index=index)["tool_calls"])
                elif item_type == "function_call_output":
                    flush_tool_calls()
                    messages.append(_function_call_output(raw_item, index=index))
                elif item_type in {"input_file", "file"}:
                    flush_tool_calls()
                    _reject(f"input[{index}]", "file inputs are not supported")
                else:
                    flush_tool_calls()
                    _reject(
                        f"input[{index}].type",
                        f"unsupported Responses input item type {item_type!r}",
                    )
            flush_tool_calls()
    else:
        _reject("input", "input must be a string or an array", code="invalid_type")
    if request.instructions:
        messages.insert(0, {"role": "system", "content": request.instructions})
    if not messages:
        _reject("input", "input must not be empty", code="invalid_value")
    return messages


def response_output_to_chat_messages(
    output: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize one terminal Responses output as one assistant turn."""

    content = ""
    reasoning = ""
    tool_calls: list[dict[str, Any]] = []
    for item in output:
        item_type = str(item.get("type") or "")
        if item_type == "message":
            for part in item.get("content") or []:
                if isinstance(part, Mapping) and part.get("type") in {
                    "output_text",
                    "text",
                }:
                    content += str(part.get("text") or "")
        elif item_type == "reasoning":
            for part in item.get("content") or []:
                if isinstance(part, Mapping):
                    reasoning += str(part.get("text") or "")
        elif item_type == "function_call":
            tool_calls.append(
                {
                    "id": str(item.get("call_id") or item.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "arguments": str(item.get("arguments") or ""),
                    },
                }
            )
    if not content and not reasoning and not tool_calls:
        return []
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return [message]


def _tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    translated: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            _reject(f"tools[{index}]", "tools entries must be objects", code="invalid_type")
        if tool.get("type") != "function":
            _reject(
                f"tools[{index}].type",
                "only function tools are supported by this Responses adapter",
                code="unsupported_tool",
            )
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            _reject(f"tools[{index}].name", "function tool name must be non-empty")
        function = {
            "name": name,
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        }
        if tool.get("description") is not None:
            function["description"] = str(tool["description"])
        if tool.get("strict") is not None:
            function["strict"] = bool(tool["strict"])
        translated.append({"type": "function", "function": function})
    return translated


def _tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice not in {"none", "auto", "required"}:
            _reject("tool_choice", f"unsupported tool_choice mode {tool_choice!r}")
        return tool_choice
    if not isinstance(tool_choice, Mapping):
        _reject("tool_choice", "tool_choice must be a string or an object", code="invalid_type")
    if tool_choice.get("type") != "function":
        _reject(
            "tool_choice.type",
            "only function tool_choice is supported by this Responses adapter",
            code="unsupported_tool",
        )
    name = tool_choice.get("name")
    if not isinstance(name, str) or not name:
        _reject("tool_choice.name", "function tool_choice name must be non-empty")
    return {"type": "function", "function": {"name": name}}


def responses_request_to_chat(request: ResponsesRequest) -> dict[str, Any]:
    """Return kwargs for the existing ``ChatCompletionRequest`` model."""

    extras = set(request.model_extra or {})
    unsupported = sorted(extras & _UNSUPPORTED_TOP_LEVEL)
    if unsupported:
        param = unsupported[0]
        _reject(param, f"{param} is not supported by the ephemeral Responses adapter")
    if extras:
        param = sorted(extras)[0]
        _reject(param, f"unknown Responses parameter {param!r}", code="unknown_parameter")
    if request.text:
        text_format = request.text.get("format")
        if text_format not in (None, {"type": "text"}):
            _reject("text.format", "structured output formats are not supported yet")
        unsupported_text = set(request.text) - {"format", "verbosity"}
        if unsupported_text:
            param = f"text.{sorted(unsupported_text)[0]}"
            _reject(param, f"{param} is not supported")
        if request.text.get("verbosity") is not None:
            _reject(
                "text.verbosity",
                "text.verbosity is not supported by the ephemeral Responses adapter",
            )

    reasoning = request.reasoning or {}
    unsupported_reasoning = set(reasoning) - {"effort", "summary"}
    if unsupported_reasoning:
        param = f"reasoning.{sorted(unsupported_reasoning)[0]}"
        _reject(param, f"{param} is not supported")
    effort = reasoning.get("effort")
    summary = reasoning.get("summary")
    if summary is not None:
        _reject(
            "reasoning.summary",
            "reasoning.summary is not supported by the ephemeral Responses adapter",
        )

    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    if reasoning:
        # "off"/"none" are thinking-disabled sentinels, not effort tiers
        # (OpenAI's own Responses vocabulary includes "none"). Passing them
        # through as reasoning_effort made the effort validator raise AFTER
        # the SSE stream had opened, severing the connection without a
        # terminal event (2026-09-28 dragon report). Resolve them here, at
        # the protocol boundary, before any event is emitted.
        if effort is None:
            enable_thinking = True
        else:
            effort_text = str(effort).strip().lower()
            if effort_text in _REASONING_DISABLED_EFFORTS:
                enable_thinking = False
            else:
                try:
                    reasoning_effort = normalize_reasoning_effort(effort_text)
                except ValueError:
                    _reject(
                        "reasoning.effort",
                        f"unsupported reasoning effort {effort!r}",
                        code="invalid_value",
                    )
                enable_thinking = True

    return {
        "model": request.model,
        "messages": _messages(request),
        "max_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "seed": request.seed,
        "stop": request.stop,
        "stream": request.stream,
        "tools": _tools(request.tools),
        "tool_choice": _tool_choice(request.tool_choice),
        "parallel_tool_calls": request.parallel_tool_calls,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "metadata": request.metadata,
        "user": request.user,
    }
