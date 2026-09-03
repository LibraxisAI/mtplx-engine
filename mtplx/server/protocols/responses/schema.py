"""Request schema for the first, deliberately ephemeral Responses surface."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ResponsesRequest(BaseModel):
    """Supported W0 subset of ``POST /v1/responses``.

    Unknown fields are retained so the translator can reject them with an
    OpenAI-shaped error naming the exact parameter. Silently swallowing a
    stateful or structured-output option would be a false compatibility claim.
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: Any = None
    instructions: str | None = None
    stream: bool = False
    store: bool | None = None
    previous_response_id: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    stop: Any = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    user: str | None = None
