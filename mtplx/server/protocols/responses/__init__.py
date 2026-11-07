"""Native, ephemeral OpenAI Responses protocol adapter."""

from .encoder import (
    response_failure_envelope,
    response_from_chat_completion,
    responses_stream_from_chat_sse,
    responses_stream_from_turn_events,
)
from .schema import ResponsesRequest
from .store import ResponseRegistry, ResponseStoreError
from .translate import (
    ResponsesProtocolError,
    response_output_to_chat_messages,
    responses_request_to_chat,
)

__all__ = [
    "ResponseRegistry",
    "ResponseStoreError",
    "ResponsesProtocolError",
    "ResponsesRequest",
    "response_failure_envelope",
    "response_from_chat_completion",
    "response_output_to_chat_messages",
    "responses_request_to_chat",
    "responses_stream_from_chat_sse",
    "responses_stream_from_turn_events",
]
