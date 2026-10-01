"""Issue #468: a bare name on the first line must not hold the stream forever.

The initial orphan guard buffers the start of a stream while it may still turn
into dangling tool-control markup (``value>``, ``value=abc>``). It classified
from the first line only, and the first line cannot grow once a newline
arrives, so an answer whose first line was exactly a bare name (``value``,
``type``, ``result`` ...) or a prefix of one stayed in ``hold`` and the whole
answer was released at ``finish()``: minutes of silence on a long stream.
"""

import mtplx.server.openai as srv

BODY = "x" * 5000


def _stream(text: str) -> tuple[str, str, srv._InitialOrphanToolControlStreamGuard]:
    guard = srv._InitialOrphanToolControlStreamGuard()
    emitted = "".join(guard.feed(char) for char in text)
    return emitted, guard.finish(), guard


def test_bare_name_first_line_streams_once_the_line_closes():
    for first in (
        "value",
        "type",
        "function",
        "parameter",
        "invoke",
        "result",
        "func",
        "tool_call",
        "toolExec",
        "value=abc",
        "valu",
        "Value",
    ):
        text = first + "\n" + BODY
        emitted, tail, guard = _stream(text)
        assert emitted == text, first
        assert tail == ""
        assert guard.suppressed is False


def test_first_line_closed_by_carriage_return_streams():
    text = "value\r\n" + BODY
    emitted, tail, _ = _stream(text)
    assert emitted == text
    assert tail == ""


def test_unrelated_first_line_still_streams():
    text = "<!DOCTYPE html>\n" + BODY
    emitted, tail, _ = _stream(text)
    assert emitted == text
    assert tail == ""


def test_bare_control_line_with_closing_bracket_is_still_suppressed():
    emitted, tail, guard = _stream("value>\n" + BODY)
    assert emitted == ""
    assert tail == ""
    assert guard.suppressed is True


def test_unclosed_first_line_still_defers_to_finish():
    emitted, tail, guard = _stream("value")
    assert emitted == ""
    assert tail == "value"
    assert guard.suppressed is False


def test_classifier_resolves_when_the_first_line_closes():
    state = srv._initial_orphan_tool_control_state
    assert state("value") == "hold"
    assert state("value=abc") == "hold"
    assert state("valu") == "hold"
    assert state("value>") == "orphan"
    assert state("value=abc>") == "orphan"
    assert state("value\n") == "normal"
    assert state("value=abc\nmore") == "normal"
    assert state("valu\n") == "normal"
    assert state("value is important") == "normal"
    assert state("<tool_call>") == "orphan"
    assert state("<tool_ca") == "hold"
    assert state("<tool_ca\n" + BODY) == "normal"
