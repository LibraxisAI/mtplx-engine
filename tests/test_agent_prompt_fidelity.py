"""Agent-prompt fidelity: the prompt the server builds for an agent session
against the chat template's own rendering, id for id.

The always-on tests run the real shipped Qwen chat-template fixture
(``Qwen36TemplateTokenizer`` from test_scoped_reasoning_history) under a
tokenizer that, like the real vocabulary, has one token for a blank line
beside the single newline: the only place where cutting the text before
encoding changes the ids. The pack tests at the end run the full audit
(scripts/audit_agent_prompt_fidelity.py) on the shipped packs and skip when a
pack is not installed.

What is pinned:
- a chat without tools is the template, id for id, in every client lane;
- a tool session is the template outside the system turn, thinking on and off;
- with thinking off the closed empty think scaffold is never cut in two
  (it used to be from the second request of every tool session on, which also
  made the first request's prompt stop being a prefix of the second);
- the postcommit prediction and the next request agree with thinking off;
- thinking-on seams are exactly where they were;
- the chain the audit calls is the chain the endpoint runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mtplx.server import openai as oa
from mtplx.server.openai import ChatMessage, _encode_messages, _postcommit_next_turn_prefix_ids
from tests.test_scoped_reasoning_history import Qwen36TemplateTokenizer
from tests.test_server_openai import ForegroundState, _fake_generation, _fake_state

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_agent_prompt_fidelity.py"
_spec = importlib.util.spec_from_file_location("audit_agent_prompt_fidelity", _SCRIPT)
audit = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = audit
_spec.loader.exec_module(audit)


class BlankLineTokenizer(Qwen36TemplateTokenizer):
    """Character ids, except that a blank line is ONE token, the way the real
    vocabulary carries '\\n\\n' (271) beside '\\n' (198)."""

    BLANK_LINE = 0x110000 + 271  # outside the Unicode range: never an ord()

    def encode(self, text, **_kwargs):
        text = str(text)
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text.startswith("\n\n", i):
                ids.append(self.BLANK_LINE)
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, tokens, **_kwargs):
        return "".join("\n\n" if int(t) == self.BLANK_LINE else chr(int(t)) for t in tokens)

    def apply_chat_template(self, messages, **kwargs):
        rendered = super().apply_chat_template(messages, **{**kwargs, "tokenize": False})
        return self.encode(rendered) if kwargs.get("tokenize") else rendered


def _reference_ids(tokenizer, messages, tools, *, thinking: bool) -> list[int]:
    """The template's own rendering, one-pass encoded. Tools are given in the
    key order the server normalizes them to, so the comparison is about the
    conversation and not about the documented schema key sort."""
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": thinking, "preserve_thinking": True}
    if tools:
        kwargs["tools"] = oa._normalize_tool_specs(tools)
    return tokenizer.encode(tokenizer.apply_chat_template(audit.Reference.template_messages(messages), **kwargs))


def _state(tokenizer):
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.has_foreground = foreground.has_foreground
    state.runtime.tokenizer = tokenizer
    state.sessions = audit._AuditSessions()
    state.args.stats_footer = False
    state.context_window = 1_000_000  # character ids: prompts are long
    return state


def _served_ids(path, messages, tools, *, lane: str, thinking: bool) -> list[int]:
    return path.prompt(messages, tools, lane=lane, thinking=thinking)["ids"]


@pytest.mark.parametrize("lane", ["pi", "opencode", "hermes", "anonymous"])
@pytest.mark.parametrize("thinking", [True, False])
def test_a_chat_without_tools_is_the_template_id_for_id(lane, thinking):
    tokenizer = BlankLineTokenizer()
    path = audit.MtplxPath(_state(tokenizer))
    session = audit.session_plain_chat()
    messages = session["messages"] if thinking else audit.without_reasoning(session["messages"])
    cuts = audit.turn_boundaries(messages)
    assert len(cuts) == 4
    for cut in cuts:
        assert _served_ids(path, messages[:cut], None, lane=lane, thinking=thinking) == _reference_ids(
            tokenizer, messages[:cut], None, thinking=thinking
        ), f"turn boundary at message {cut}"


@pytest.mark.parametrize("build", [audit.session_odd_whitespace, audit.session_multilingual])
def test_a_thinking_tool_session_is_the_template_outside_the_system_turn(build):
    # Native tool-prompt mode: no contract text is added, so with the tools in
    # normalized key order the whole prompt must be the template's.
    tokenizer = BlankLineTokenizer()
    path = audit.MtplxPath(_state(tokenizer))
    session = build()
    for cut in audit.turn_boundaries(session["messages"]):
        messages = session["messages"][:cut]
        assert _served_ids(path, messages, session["tools"], lane="native", thinking=True) == _reference_ids(
            tokenizer, messages, session["tools"], thinking=True
        ), f"turn boundary at message {cut}"


@pytest.mark.parametrize("lane", ["native", "hermes", "pi"])
def test_thinking_off_never_cuts_the_closed_think_scaffold(lane):
    # A thinking-off prompt carries the whole closed scaffold
    # '<think>\n\n</think>\n\n'; no generation ever began after '<think>\n'.
    # From the second request of a tool session on, the history encoder cut
    # there anyway: '\n\n' became '\n','\n' in every history turn AND in the
    # generation prompt the model continues from.
    tokenizer = BlankLineTokenizer()
    path = audit.MtplxPath(_state(tokenizer))
    session = audit.session_hermes_loop()
    scaffold = tokenizer.encode("<think>\n\n</think>\n\n")
    assert scaffold.count(tokenizer.BLANK_LINE) == 2
    previous: list[int] | None = None
    for cut in audit.turn_boundaries(session["messages"]):
        served = _served_ids(path, session["messages"][:cut], session["tools"], lane=lane, thinking=False)
        assert served[-len(scaffold):] == scaffold, "the generation prompt ends with the scaffold as the template encodes it"
        text = tokenizer.decode(served)
        assert _count_sublist(served, scaffold) == text.count("<think>\n\n</think>\n\n")
        if lane == "native":
            assert served == _reference_ids(tokenizer, session["messages"][:cut], session["tools"], thinking=False)
        if previous is not None:
            # What the model generated from stays a prefix of what it sees next.
            assert served[: len(previous)] == previous
        previous = served


def _count_sublist(haystack: list[int], needle: list[int]) -> int:
    return sum(1 for i in range(len(haystack) - len(needle) + 1) if haystack[i: i + len(needle)] == needle)


def test_thinking_off_postcommit_prediction_is_a_prefix_of_the_next_request():
    tokenizer = BlankLineTokenizer()
    session = audit.session_hermes_loop()
    tools = oa._normalize_tool_specs(session["tools"])
    messages = [ChatMessage(**m) for m in session["messages"]]
    assistant_turns = [i for i, m in enumerate(messages) if m.role == "assistant"]
    assert len(assistant_turns) == 3  # two tool-call turns and one plain answer
    for index in assistant_turns:
        history = messages[: index + 1]
        predicted = _postcommit_next_turn_prefix_ids(
            tokenizer,
            history,
            enable_thinking=False,
            strip_assistant_reasoning_history=False,
            preserve_reasoning_history=True,
            tools=tools,
            assistant_tool_calls=history[-1].tool_calls,
            tool_prompt_mode="native",
        )
        next_prompt = _encode_messages(
            tokenizer,
            messages[: index + 2],
            enable_thinking=False,
            preserve_reasoning_history=True,
            tools=tools,
            tool_prompt_mode="native",
        )
        assert predicted, f"assistant turn at message {index}"
        assert next_prompt[: len(predicted)] == predicted, f"assistant turn at message {index}"


def test_thinking_on_generation_seams_are_where_they_were():
    # Thinking on, a history turn with an empty think block (no reasoning was
    # echoed): the prompt that turn was generated from ended with '<think>\n',
    # so the encode stays cut there. One-pass encoding merges the blank line.
    tokenizer = BlankLineTokenizer()
    path = audit.MtplxPath(_state(tokenizer))
    session = audit.session_hermes_loop()
    cuts = audit.turn_boundaries(session["messages"])
    first = _served_ids(path, session["messages"][: cuts[0]], session["tools"], lane="native", thinking=True)
    second = _served_ids(path, session["messages"][: cuts[1]], session["tools"], lane="native", thinking=True)
    assert second[: len(first)] == first
    seam = [ord(c) for c in "<think>"] + [ord("\n"), ord("\n")] + [ord(c) for c in "</think>"]
    assert _count_sublist(second, seam) == 1
    reference = _reference_ids(tokenizer, session["messages"][: cuts[1]], session["tools"], thinking=True)
    assert tokenizer.decode(second) == tokenizer.decode(reference)
    assert second != reference  # the documented, generation-time seam


@pytest.mark.parametrize("lane", ["pi", "opencode", "hermes"])
@pytest.mark.parametrize("thinking", [True, False])
def test_the_audit_chain_is_the_endpoint_chain(monkeypatch, lane, thinking):
    tokenizer = BlankLineTokenizer()
    state = _state(tokenizer)
    captured: list[list[int]] = []

    def fake_run_generation(_state, prompt_ids, **_kwargs):
        captured.append([int(t) for t in prompt_ids])
        return _fake_generation("ok")

    monkeypatch.setattr(oa, "_run_generation", fake_run_generation)
    client = TestClient(oa.create_app(state))
    path = audit.MtplxPath(state)
    session = audit.session_hermes_loop() if lane == "hermes" else audit.session_opencode()
    headers = {**audit.LANES[lane]["headers"], "x-mtplx-cache-mode": "bypass"}
    for cut in audit.turn_boundaries(session["messages"])[:4]:
        messages = session["messages"][:cut] if thinking else audit.without_reasoning(session["messages"][:cut])
        state.args.enable_thinking = thinking
        response = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": state.model_id, "messages": messages, "tools": session["tools"]},
        )
        assert response.status_code == 200, response.text
        expected = path.prompt(messages, session["tools"], lane=lane, thinking=thinking, headers=headers)["ids"]
        assert captured[-1] == expected, f"turn boundary at message {cut}"


def _opencode_turn_with_a_committed_seam(tokenizer, path, *, seam_in: str):
    """One OpenCode turn whose committed stream carries the model's own
    non-canonical seam (a blank line emitted as two newline tokens) in its
    reasoning or in its visible text. Returns (messages, tools, committed)."""
    lane = "opencode"
    tools = audit.opencode_tools()
    builder = audit._Builder("You are opencode.")
    builder.user("Add a --json flag to the export command.")
    reasoning = "Find the file.\n\nThen read it." if seam_in == "reasoning" else "Find the file, then read it."
    visible = "I'll look first.\n\nThen edit." if seam_in == "visible_text" else "I'll look first."
    builder.assistant(visible, reasoning=reasoning, calls=[("glob", {"pattern": "src/**/export*.ts"})])
    builder.tool("/work/src/commands/export.ts")
    messages = builder.messages
    first = path.prompt(messages[:2], tools, lane=lane, thinking=True, use_session=True)
    render = lambda upto, generation: tokenizer.apply_chat_template(  # noqa: E731
        audit.Reference.template_messages(messages[:upto]), tokenize=False, add_generation_prompt=generation,
        enable_thinking=True, preserve_thinking=True,
    )
    generated_text = render(3, False)[len(render(2, True)):]
    assert generated_text.endswith("<|im_end|>\n")
    generated_ids = tokenizer.encode(generated_text[: -len("<|im_end|>\n")])
    before_seam = {"reasoning": "Find the file.", "after_think_close": "</think>", "visible_text": "look first."}[seam_in]
    seam_at = next(
        i for i, token in enumerate(generated_ids)
        if token == tokenizer.BLANK_LINE and tokenizer.decode(generated_ids[:i]).endswith(before_seam)
    )
    generated_ids[seam_at: seam_at + 1] = [ord("\n"), ord("\n")]
    assert tokenizer.decode(generated_ids) + "<|im_end|>\n" == generated_text
    committed = list(first["ids"]) + generated_ids
    path.sessions.committed[audit.LANES[lane]["headers"]["x-mtplx-session-id"]] = tuple(committed)
    return messages, tools, committed


@pytest.mark.parametrize("seam_in", ["reasoning", "after_think_close", "visible_text"])
def test_a_seam_the_model_left_does_not_cost_the_committed_turn_body(seam_in):
    # OpenCode echoes the reasoning and never the text before a tool call, so
    # the raw prompt and the canonical one (committed think + committed body)
    # share the reasoning and differ in the body. A seam inside the reasoning
    # (or right after '</think>') stops BOTH at the same token; the gate used
    # to read that tie as "the canonical encode is no better" before either
    # had been spliced, served the raw prompt, and the turn's body was
    # re-prefilled without the text the model had written. A seam inside the
    # visible text never tied (the raw prompt parts earlier) and is the
    # unchanged control.
    tokenizer = BlankLineTokenizer()
    path = audit.MtplxPath(_state(tokenizer))
    messages, tools, committed = _opencode_turn_with_a_committed_seam(tokenizer, path, seam_in=seam_in)
    served = path.prompt(messages, tools, lane="opencode", thinking=True, use_session=True)
    assert served["ids"][: len(committed)] == committed
    assert "I'll look first." in tokenizer.decode(served["ids"])
    outcome = served["canonicalization"]
    assert outcome["applied"] is True and outcome["turns_substituted"] == 1
    assert outcome["cp_spliced"] == len(committed)


# ---------------------------------------------------------------------------
# The shipped packs (tokenizer and template only; skipped when not installed)
# ---------------------------------------------------------------------------
_PACK_ROOT = Path.home() / ".mtplx" / "models"


@pytest.fixture(scope="module", params=audit.DEFAULT_PACKS)
def pack_audit(request):
    pack = _PACK_ROOT / request.param
    if not (pack / "tokenizer.json").exists() or not (pack / "config.json").exists():
        pytest.skip(f"{request.param} is not installed")
    return audit.Reference(pack), audit.MtplxPath.for_pack(pack)


def _stateless_record(ref, path, session, *, lane: str, thinking_mode: str, cut: int):
    thinking, keep_reasoning = audit.THINKING_MODES[thinking_mode]
    messages = session["messages"] if keep_reasoning else audit.without_reasoning(session["messages"])
    served = path.prompt(messages[:cut], session["tools"], lane=lane, thinking=thinking)
    text = ref.render(messages[:cut], session["tools"], enable_thinking=served["thinking"],
                      reasoning_effort=served["reasoning_effort"])
    return audit.compare(ref, ref.encode(text), served["ids"], thinking=thinking, lane=lane)


def test_pack_server_tokenizer_agrees_with_the_pack_declared_tokenization(pack_audit):
    ref, path = pack_audit
    for session in (audit.session_multilingual(), audit.session_odd_whitespace()):
        for message in session["messages"]:
            for key in ("content", "reasoning_content"):
                text = message.get(key)
                if isinstance(text, str) and text:
                    assert [int(t) for t in oa._encode_rendered_chat_text(path.tokenizer, text)] == ref.encode(text)


@pytest.mark.parametrize("thinking_mode", ["on", "off"])
def test_pack_chat_without_tools_is_identical(pack_audit, thinking_mode):
    ref, path = pack_audit
    session = audit.session_plain_chat()
    for lane in audit.SHAPE_LANES["plain"]:
        for cut in audit.turn_boundaries(session["messages"]):
            record = _stateless_record(ref, path, session, lane=lane, thinking_mode=thinking_mode, cut=cut)
            assert record["classification"] == "identical", (lane, cut, record)


@pytest.mark.parametrize("thinking_mode", ["on", "off", "off_after_on"])
def test_pack_tool_sessions_have_no_unexplained_difference(pack_audit, thinking_mode):
    ref, path = pack_audit
    for session in (audit.session_hermes_loop(), audit.session_multilingual(), audit.session_odd_whitespace()):
        if thinking_mode == "off_after_on" and not any(m.get("reasoning_content") for m in session["messages"]):
            continue
        for lane in audit.SHAPE_LANES[session["shape"]]:
            for cut in audit.turn_boundaries(session["messages"]):
                record = _stateless_record(ref, path, session, lane=lane, thinking_mode=thinking_mode, cut=cut)
                assert record["classification"] != "UNEXPLAINED", (session["name"], lane, cut, record.get("unexplained"))


@pytest.mark.parametrize("thinking_mode", ["on", "off"])
def test_pack_conversation_body_is_identical_outside_the_system_turn(pack_audit, thinking_mode):
    # Arabic/CJK/emoji text and CRLF/tab/trailing-space code, tool calls with
    # nested arguments, tool results with leading and trailing blank lines:
    # every user, assistant and tool turn and the generation prompt equal the
    # template's ids. Only the system turn differs (tool contract, key sort).
    ref, path = pack_audit
    for session in (audit.session_multilingual(), audit.session_odd_whitespace()):
        for cut in audit.turn_boundaries(session["messages"]):
            record = _stateless_record(ref, path, session, lane="pi", thinking_mode=thinking_mode, cut=cut)
            assert record["body_identical"], (session["name"], cut, record["mechanisms"], record.get("unexplained"))


def test_pack_thinking_off_tool_session_extends_its_own_committed_stream_unaided(pack_audit):
    # With canonical generated ids the next prompt must extend prompt+generated
    # as encoded, with no splice doing repair work.
    ref, path = pack_audit
    rows = audit.splice_scenario(ref, path, audit.session_hermes_loop(), lane="hermes", thinking_mode="off", mode="canonical")
    assert len(rows) == 4
    for row in rows[1:]:
        assert row["extends_committed"], row
        assert row["raw_equals_served"], row
