"""A thinking block ends at ``</think>`` or at the start of a real tool call.

Founder report, 2026-09-21: a follow-up message pasted a traceback that
contained ``</parameter>``. The model quoted the tag a few words into its
thinking, the stream splitter took the quote for tool-call markup, left the
thinking state, and streamed the rest of the thinking block to the visible
chat (chat store: 39 and 27 characters of reasoning, visible text starting
exactly at the tag; stop and resend reproduced it every time).

The contract these tests pin:

* inside thinking, only a tool-call OPENER can end the block, and only where
  a call can really begin: tools were declared, the opener starts a line, the
  reasoning is not inside a code fence, and the opener looks like a call;
* a closing tag, or ``<parameter=...>``, can never begin a call;
* a marker the model merely quotes stays reasoning, however it is chunked.
"""

from __future__ import annotations

import random

import pytest

from mtplx.server import openai as oa

ANSWER = "Here is the fixed file."

QUOTED_MARKERS = (
    "<tool_call>",
    "<function=read_file>",
    "<parameter=path>",
    "</parameter>",
    "</function>",
    "</tool_call>",
)

REAL_CALL = (
    "<tool_call>\n<function=read_file>\n"
    "<parameter=path>\nsrc/app.py\n</parameter>\n"
    "</function>\n</tool_call>"
)


def _splitter(*, tools: bool, tool_names=("read_file", "write")):
    kwargs = dict(
        thinking_enabled=True,
        recover_unclosed_reasoning_as_content=False,
        start_inside_thinking=True,
        suppress_orphan_tool_markup=not tools,
        trim_visible_content_edges=True,
    )
    if tools:
        kwargs["tool_names"] = tool_names
    return oa._ThinkingContentStreamSplitter(**kwargs)


def _pieces(text: str, mode: str) -> list[str]:
    if mode == "whole":
        return [text]
    if mode == "chars":
        return list(text)
    rng = random.Random(f"{mode}:{len(text)}")
    pieces, index = [], 0
    while index < len(text):
        step = rng.randint(1, 11)
        pieces.append(text[index : index + step])
        index += step
    return pieces


def _drive(splitter, text: str, mode: str = "whole"):
    chunks = list(splitter.start())
    for piece in _pieces(text, mode):
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish())
    reasoning = "".join(t for field, t in chunks if field == "reasoning_content")
    content = "".join(t for field, t in chunks if field == "content")
    return reasoning, content


CHUNKINGS = ("whole", "chars", "random-a", "random-b")


@pytest.mark.parametrize("mode", CHUNKINGS)
@pytest.mark.parametrize("tools", (False, True))
@pytest.mark.parametrize("marker", QUOTED_MARKERS)
def test_marker_quoted_in_backticks_stays_reasoning(marker, tools, mode):
    thinking = (
        f"The traceback shows a stray `{marker}` tag on line 1785, "
        "so the file ended badly. I should rewrite it."
    )
    reasoning, content = _drive(
        _splitter(tools=tools), f"{thinking}</think>\n\n{ANSWER}", mode
    )
    assert reasoning == thinking
    assert content == ANSWER


@pytest.mark.parametrize("mode", CHUNKINGS)
@pytest.mark.parametrize("tools", (False, True))
def test_founder_traceback_case(tools, mode):
    thinking = (
        "The user pasted:\n"
        '  File "pyrunner.py", line 1785\n'
        "    </parameter>\n"
        "    ^\n"
        "SyntaxError: invalid syntax\n"
        "So a `</parameter>` tag leaked into the code. Remove that line."
    )
    reasoning, content = _drive(
        _splitter(tools=tools), f"{thinking}</think>\n\n{ANSWER}", mode
    )
    assert reasoning == thinking
    assert content == ANSWER


@pytest.mark.parametrize("mode", CHUNKINGS)
@pytest.mark.parametrize("tools", (False, True))
@pytest.mark.parametrize("closer", ("</parameter>", "</function>", "</tool_call>"))
def test_closing_tag_on_its_own_line_never_ends_thinking(closer, tools, mode):
    thinking = f"The bad line is:\n{closer}\nand nothing else is wrong."
    reasoning, content = _drive(
        _splitter(tools=tools), f"{thinking}</think>\n\n{ANSWER}", mode
    )
    assert reasoning == thinking
    assert content == ANSWER


@pytest.mark.parametrize("mode", CHUNKINGS)
def test_parameter_opener_on_its_own_line_never_ends_thinking(mode):
    thinking = "The argument block starts with:\n<parameter=path>\nwhich is fine."
    reasoning, content = _drive(
        _splitter(tools=True), f"{thinking}</think>\n\n{ANSWER}", mode
    )
    assert reasoning == thinking
    assert content == ANSWER


@pytest.mark.parametrize("mode", CHUNKINGS)
def test_opener_at_line_start_that_is_not_a_call_stays_reasoning(mode):
    thinking = (
        "Notes on the protocol.\n"
        "<tool_call> is the wrapper the template expects, and\n"
        "<function=delete_everything> names no tool this request declared."
    )
    reasoning, content = _drive(
        _splitter(tools=True), f"{thinking}</think>\n\n{ANSWER}", mode
    )
    assert reasoning == thinking
    assert content == ANSWER


@pytest.mark.parametrize("mode", CHUNKINGS)
def test_call_shown_inside_a_code_fence_stays_reasoning(mode):
    thinking = (
        "The format I must produce later is:\n"
        f"```xml\n{REAL_CALL}\n```\n"
        "First I need to decide which file to read."
    )
    reasoning, content = _drive(
        _splitter(tools=True), f"{thinking}</think>\n\n{ANSWER}", mode
    )
    assert reasoning == thinking
    assert content == ANSWER


@pytest.mark.parametrize("mode", CHUNKINGS)
def test_no_declared_tools_means_no_marker_ends_thinking(mode):
    thinking = f"I cannot call tools here, but the call would be:\n{REAL_CALL}\nSo I answer in prose."
    reasoning, content = _drive(
        _splitter(tools=False), f"{thinking}</think>\n\n{ANSWER}", mode
    )
    assert reasoning == thinking
    assert content == ANSWER


@pytest.mark.parametrize("mode", CHUNKINGS)
@pytest.mark.parametrize("tool_names", (("read_file", "write"), None))
def test_real_call_without_think_close_still_ends_thinking(tool_names, mode):
    """F40 stays: with tools declared the model may start a call without
    closing its think block. ``tool_names=None`` is the legacy construction
    (names unknown), where the call's shape alone decides."""
    splitter = _splitter(tools=True, tool_names=tool_names)
    reasoning, content = _drive(splitter, "Let me read the file.\n" + REAL_CALL, mode)
    assert reasoning == "Let me read the file.\n"
    assert content.startswith("<tool_call>")
    assert "<function=read_file>" in content
    assert splitter.tool_preamble_recovered_content == "Let me read the file."


@pytest.mark.parametrize("mode", CHUNKINGS)
def test_bare_function_opener_is_a_call_when_the_tool_is_declared(mode):
    call = "<function=read_file>\n<parameter=path>\nsrc/app.py\n</parameter>\n</function>"
    splitter = _splitter(tools=True)
    reasoning, content = _drive(splitter, "Reading it now.\n" + call, mode)
    assert reasoning == "Reading it now.\n"
    assert content.startswith("<function=read_file>")


@pytest.mark.parametrize("mode", CHUNKINGS)
def test_json_style_call_still_ends_thinking(mode):
    call = '<tool_call>\n{"name": "read_file", "arguments": {"path": "src/app.py"}}\n</tool_call>'
    splitter = _splitter(tools=True)
    reasoning, content = _drive(splitter, "Reading it now.\n" + call, mode)
    assert reasoning == "Reading it now.\n"
    assert content.startswith("<tool_call>")


@pytest.mark.parametrize("mode", CHUNKINGS)
def test_quote_then_real_call_in_one_turn(mode):
    thinking = "The error names `</parameter>`. I will read the file first.\n"
    splitter = _splitter(tools=True)
    reasoning, content = _drive(splitter, thinking + REAL_CALL, mode)
    assert reasoning == thinking
    assert content.startswith("<tool_call>")


def test_server_factory_hands_the_declared_tool_names_to_the_splitter(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(oa, "_reasoning_parser_for_state", lambda _state: "qwen3")
    splitter = oa._stream_splitter_for_state(
        SimpleNamespace(),
        thinking_enabled=True,
        recover_unclosed_reasoning_as_content=False,
        suppress_orphan_tool_markup=False,
        tool_names=["read_file", "bash"],
    )
    reasoning, content = _drive(
        splitter,
        "Two notes.\n<function=delete_everything> is not a tool here.\n"
        "<function=Shell>\n<parameter=command>\nls\n</parameter>\n</function>",
    )
    # "Shell" is how the model sometimes spells the declared "bash" tool; the
    # splitter uses the same name mapping as the tool-call parser.
    assert reasoning == "Two notes.\n<function=delete_everything> is not a tool here.\n"
    assert content.startswith("<function=Shell>")
