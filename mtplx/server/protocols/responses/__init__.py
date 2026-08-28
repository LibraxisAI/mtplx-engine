"""Native, ephemeral OpenAI Responses protocol adapter."""

from .encoder import response_from_chat_completion, responses_stream_from_chat_sse
from .schema import ResponsesRequest
from .translate import ResponsesProtocolError, responses_request_to_chat

__all__ = [
    "ResponsesProtocolError",
    "ResponsesRequest",
    "response_from_chat_completion",
    "responses_request_to_chat",
    "responses_stream_from_chat_sse",
]
