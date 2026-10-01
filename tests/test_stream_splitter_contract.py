"""Every reasoning stream splitter satisfies the contract the server reads.

Issue #517 (MTPLX 2.11.3): every streamed Gemma 4 reply without tools failed
after a successful generation with

    'Gemma4ThinkingContentStreamSplitter' object has no attribute
    'suppressed_tool_markup_chars'

The stream lane reads a fixed set of members from whatever splitter
``_stream_splitter_for_state`` returns. The Qwen splitter inside the server
module defined all of them; the codec-module splitters (Gemma 4 is the one the
factory routes there today) did not, so the attribute read at the end of a
no-tools turn raised. The contract now lives on the codec base class, and
these tests hold every registered parser to it through the server's own
factory, so the next codec cannot repeat the failure.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx import reasoning_codecs as rc
from mtplx.server import openai as oa

# Every parser id the server can be configured with, plus an unknown one.
PARSER_IDS = ("qwen3", "gemma4", "lfm2", "none", "something-new")

# The members the stream lane reads from a splitter (mtplx/server/openai.py).
CONTRACT_ATTRIBUTES = (
    "reentry_count",
    "suppressed_tool_markup_chars",
    "tool_preamble_recovered_content",
)
CONTRACT_METHODS = ("start", "feed", "finish")


def _server_splitter(monkeypatch, parser: str, *, thinking: bool, tools: bool):
    monkeypatch.setattr(oa, "_reasoning_parser_for_state", lambda _state: parser)
    return oa._stream_splitter_for_state(
        SimpleNamespace(),
        thinking_enabled=thinking,
        recover_unclosed_reasoning_as_content=False,
        start_inside_thinking=True,
        suppress_orphan_tool_markup=not tools,
    )


@pytest.mark.parametrize("tools", (False, True))
@pytest.mark.parametrize("thinking", (False, True))
@pytest.mark.parametrize("parser", PARSER_IDS)
def test_factory_splitters_expose_the_whole_contract(monkeypatch, parser, thinking, tools):
    splitter = _server_splitter(monkeypatch, parser, thinking=thinking, tools=tools)
    for name in CONTRACT_METHODS:
        assert callable(getattr(splitter, name)), name
    for name in CONTRACT_ATTRIBUTES:
        assert hasattr(splitter, name), name
    assert splitter.suppressed_tool_markup_chars == 0
    assert splitter.tool_preamble_recovered_content is None
    assert splitter.reentry_count == 0


@pytest.mark.parametrize("parser", PARSER_IDS)
def test_codec_module_splitters_expose_the_whole_contract(parser):
    splitter = rc.stream_splitter_for_parser(parser, thinking_enabled=True)
    assert isinstance(splitter, rc.ReasoningContentStreamSplitter)
    for name in CONTRACT_ATTRIBUTES:
        assert hasattr(splitter, name), name
    assert splitter.suppressed_tool_markup_chars == 0
    assert splitter.tool_preamble_recovered_content is None


@pytest.mark.parametrize("recover", (False, True))
@pytest.mark.parametrize("parser", PARSER_IDS)
def test_finish_accepts_the_recovery_keyword_everywhere(monkeypatch, parser, recover):
    """One ``finish`` signature: the server passes the keyword to every codec
    and needs no ``TypeError`` fallback to discover which ones take it."""
    splitter = _server_splitter(monkeypatch, parser, thinking=True, tools=False)
    splitter.start()
    splitter.feed("plain visible text")
    chunks = splitter.finish(recover_unclosed_reasoning_as_content=recover)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, tuple) and len(chunk) == 2 for chunk in chunks)


def test_finish_helper_no_longer_swallows_type_errors():
    class Broken:
        def finish(self, *, recover_unclosed_reasoning_as_content=None):
            raise TypeError("a real bug inside finish")

    with pytest.raises(TypeError, match="a real bug inside finish"):
        oa._finish_stream_splitter(Broken(), recover_unclosed_reasoning=False)


def test_gemma4_no_tools_turn_reads_the_counter_the_way_the_stream_lane_does(monkeypatch):
    """The exact read that failed in 2.11.3, on the exact object."""
    splitter = _server_splitter(monkeypatch, "gemma4", thinking=True, tools=False)
    out = list(splitter.start())
    for piece in (
        rc.GEMMA4_THINK_OPEN,
        "Gelfond's constant is e to the pi.",
        rc.GEMMA4_THINK_CLOSE,
        "23.1406926",
    ):
        out.extend(splitter.feed(piece))
    out.extend(oa._finish_stream_splitter(splitter, recover_unclosed_reasoning=False))
    assert not (splitter.suppressed_tool_markup_chars > 0)
    content = "".join(text for field, text in out if field == "content")
    assert "23.1406926" in content
