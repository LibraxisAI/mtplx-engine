"""The committed-id splice: a model's sampled tokens are not always the
canonical BPE encoding of their own text (Hermes receipt 2026-09-08: ``"Nothing``
as one token where the tokenizer encodes ``"`` + ``Nothing``), so a client's
re-tokenized history diverges from the committed stream on identical bytes.
Wherever the two decode to the same text the committed ids are served.
"""

from __future__ import annotations

from types import SimpleNamespace

from mtplx.server import openai as oa
from mtplx.server.openai import ChatMessage, _store_generation_final_history_snapshot

from tests.test_openai_bridge import _final_state, _postcommit_state


class VocabTokenizer:
    def __init__(self, vocab: dict[int, str], *, whole: tuple[list[int], str] | None = None):
        self.vocab = dict(vocab)
        self.whole = whole

    def decode(self, ids):
        ids = [int(i) for i in ids]
        if self.whole is not None and ids == self.whole[0]:
            return self.whole[1]
        return "".join(self.vocab.get(i, f"<{i}>") for i in ids)


QUOTE, NOTHING, QUOTE_NOTHING = 10, 11, 20
VOCAB = {QUOTE: '"', NOTHING: "Nothing", QUOTE_NOTHING: '"Nothing', 30: "A", 31: "B", 32: "AB", 40: "�", 41: "�"}


def test_a_single_non_canonical_spot_is_spliced_to_the_committed_ids():
    tok = VocabTokenizer(VOCAB)
    prompt = [1, 2, 3, QUOTE, NOTHING, 5, 6]
    committed = [1, 2, 3, QUOTE_NOTHING, 5, 6]
    spliced, receipt = oa._splice_committed_token_ids(prompt, committed, tok)
    assert spliced == committed
    assert receipt["spans"] == 1
    assert receipt["tokens_in"] == 2 and receipt["tokens_out"] == 1
    assert receipt["first_divergence"] == 3 and receipt["cp_after"] == 6


def test_a_real_edit_ends_the_splice_and_keeps_the_prompt():
    tok = VocabTokenizer({**VOCAB, 12: "Nothing!"})
    prompt = [1, 2, 3, QUOTE, 12, 5, 6]
    committed = [1, 2, 3, QUOTE_NOTHING, 5, 6]
    spliced, receipt = oa._splice_committed_token_ids(prompt, committed, tok)
    assert spliced == prompt
    assert receipt["spans"] == 0


def test_several_spots_and_a_longer_prompt_keep_the_tail():
    tok = VocabTokenizer(VOCAB)
    prompt = [1, QUOTE, NOTHING, 2, 30, 31, 3, 7, 8, 9]
    committed = [1, QUOTE_NOTHING, 2, 32, 3]
    spliced, receipt = oa._splice_committed_token_ids(prompt, committed, tok)
    assert spliced == committed + [7, 8, 9]
    assert receipt["spans"] == 2


def test_split_multibyte_windows_never_match():
    tok = VocabTokenizer(VOCAB)
    prompt = [1, 40, 5]
    committed = [1, 41, 5]
    spliced, receipt = oa._splice_committed_token_ids(prompt, committed, tok)
    assert spliced == prompt and receipt["spans"] == 0


def test_the_window_bounds_the_search():
    vocab = {i: "x" for i in range(100, 110)}
    vocab[200] = "x" * 10
    tok = VocabTokenizer(vocab)
    prompt = [1] + list(range(100, 110)) + [5]
    committed = [1, 200, 5]
    spliced, receipt = oa._splice_committed_token_ids(prompt, committed, tok, window=8)
    assert spliced == prompt and receipt["spans"] == 0
    spliced, receipt = oa._splice_committed_token_ids(prompt, committed, tok, window=10)
    assert spliced == committed and receipt["spans"] == 1


NL, COLON, IM_END, WORD, OTHER = 198, 25, 248046, 300, 301
WS_VOCAB = {COLON: ":", NL: "\n", IM_END: "<|im_end|>", WORD: "word", OTHER: "other"}


def test_trailing_whitespace_the_response_stripped_is_put_back():
    # A length-cut turn ended in a newline token; the visible content came
    # back stripped, so the echoed history puts the end-of-turn marker where
    # the newline was. The committed newline is served, then the marker.
    tok = VocabTokenizer(WS_VOCAB)
    committed = [WORD, COLON, NL]
    prompt = [WORD, COLON, IM_END, NL, WORD]
    out, receipt = oa._splice_committed_token_ids(prompt, committed, tok)
    assert out == [WORD, COLON, NL, IM_END, NL, WORD]
    assert receipt["whitespace_tokens"] == 1 and receipt["spans"] == 1
    assert receipt["cp_after"] == len(committed)


def test_leading_whitespace_the_response_stripped_is_put_back():
    tok = VocabTokenizer(WS_VOCAB)
    committed = [COLON, NL, NL, WORD, COLON]
    prompt = [COLON, WORD, COLON, IM_END]
    out, receipt = oa._splice_committed_token_ids(prompt, committed, tok)
    assert out == [COLON, NL, NL, WORD, COLON, IM_END]
    assert receipt["whitespace_tokens"] == 2
    assert receipt["cp_after"] == len(committed)


def test_whitespace_before_a_real_edit_is_not_spliced():
    tok = VocabTokenizer(WS_VOCAB)
    committed = [WORD, NL, OTHER]
    prompt = [WORD, COLON, IM_END]
    out, receipt = oa._splice_committed_token_ids(prompt, committed, tok)
    assert out == prompt and receipt["spans"] == 0


def test_already_extending_prompts_are_untouched():
    tok = VocabTokenizer(VOCAB)
    committed = [1, 2, 3]
    spliced, receipt = oa._splice_committed_token_ids([1, 2, 3, 4], committed, tok)
    assert spliced == [1, 2, 3, 4] and receipt["spans"] == 0


def _gate_state(committed, committed_text, vocab):
    session = SimpleNamespace(committed_token_ids=tuple(committed))
    sessions = SimpleNamespace(
        resolve_session_id=lambda **kw: ("s1", "header.x-mtplx-session-id"),
        peek=lambda sid: session if sid == "s1" else None,
    )
    tok = VocabTokenizer(vocab, whole=(list(committed), committed_text))
    return SimpleNamespace(
        args=SimpleNamespace(strip_assistant_reasoning_history=False),
        sessions=sessions,
        runtime=SimpleNamespace(tokenizer=tok),
    )


def test_gate_serves_spliced_ids_for_a_plain_turn(monkeypatch):
    """No think interior to substitute (a reasoning-off turn), yet the resent
    history carries one non-canonical spot inside the committed answer."""
    committed = list(range(100, 200))
    text = (
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nplain answer<|im_end|>\n"
    )
    vocab = {110: "AB", 901: "A", 902: "B"}
    state = _gate_state(committed, text, vocab)
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "plain answer"},
        {"role": "user", "content": "next"},
    ]
    request = oa.ChatCompletionRequest(model="m", messages=messages)
    prompt_ids = committed[:10] + [901, 902] + committed[11:]
    observability: dict = {}
    result = oa._maybe_canonicalize_committed_reasoning(
        state,
        messages=request.messages,
        prompt_ids=prompt_ids,
        headers={},
        metadata={},
        request=request,
        thinking_enabled=True,
        reasoning_effort="xhigh",
        tools=None,
        tool_choice=None,
        tool_prompt_mode="hybrid",
        template_observability={},
        request_observability=observability,
    )
    assert result is not None
    _messages, served = result
    assert served == committed
    outcome = observability["committed_reasoning_canonicalization"]
    assert outcome["applied"] is True
    assert outcome["token_splice"]["spans"] == 1
    assert outcome["cp_spliced"] == len(committed)


def test_gate_splices_with_thinking_off(monkeypatch):
    """A reasoning-off daemon has no think interior to put back, but the
    model's own seams still exist (a length-cut turn, a non-canonical split
    in a code line); the splice runs and the receipt names the declined
    substitution."""
    committed = list(range(100, 200))
    text = (
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nplain answer<|im_end|>\n"
    )
    vocab = {110: "AB", 901: "A", 902: "B"}
    state = _gate_state(committed, text, vocab)
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "plain answer"},
        {"role": "user", "content": "next"},
    ]
    request = oa.ChatCompletionRequest(model="m", messages=messages)
    prompt_ids = committed[:10] + [901, 902] + committed[11:]
    observability: dict = {}
    result = oa._maybe_canonicalize_committed_reasoning(
        state,
        messages=request.messages,
        prompt_ids=prompt_ids,
        headers={},
        metadata={},
        request=request,
        thinking_enabled=False,
        reasoning_effort=None,
        tools=None,
        tool_choice=None,
        tool_prompt_mode="hybrid",
        template_observability={},
        request_observability=observability,
    )
    assert result is not None
    _messages, served = result
    assert served == committed
    outcome = observability["committed_reasoning_canonicalization"]
    assert outcome["declined"] == "thinking_disabled"
    assert outcome["applied"] is True
    assert outcome["token_splice"]["spans"] == 1
    # switched off, the gate declines exactly as before
    monkeypatch.setenv("MTPLX_COMMITTED_TOKEN_SPLICE", "0")
    observability2: dict = {}
    result2 = oa._maybe_canonicalize_committed_reasoning(
        state,
        messages=request.messages,
        prompt_ids=prompt_ids,
        headers={},
        metadata={},
        request=request,
        thinking_enabled=False,
        reasoning_effort=None,
        tools=None,
        tool_choice=None,
        tool_prompt_mode="hybrid",
        template_observability={},
        request_observability=observability2,
    )
    assert result2 is None
    assert observability2["committed_reasoning_canonicalization"] == {
        "applied": False,
        "declined": "thinking_disabled",
    }


def test_gate_splice_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("MTPLX_COMMITTED_TOKEN_SPLICE", "0")
    committed = list(range(100, 200))
    text = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nplain<|im_end|>\n"
    state = _gate_state(committed, text, {110: "AB", 901: "A", 902: "B"})
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    request = oa.ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "plain"}, {"role": "user", "content": "next"}])
    result = oa._maybe_canonicalize_committed_reasoning(
        state, messages=request.messages, prompt_ids=committed[:10] + [901, 902] + committed[11:],
        headers={}, metadata={}, request=request, thinking_enabled=True, reasoning_effort="xhigh",
        tools=None, tool_choice=None, tool_prompt_mode="hybrid", template_observability={}, request_observability={},
    )
    assert result is None


def test_generation_final_snapshot_accepts_a_history_that_splices_onto_the_stream(monkeypatch):
    prompt_ids = [1, 2, 3]
    generated = [4, QUOTE_NOTHING, 5, 6]
    final = prompt_ids + generated
    # The retokenized history splits the model's token and adds the turn's
    # trailing newline, as a real next-turn render does.
    history = [1, 2, 3, 4, QUOTE, NOTHING, 5, 6, 99]
    monkeypatch.setattr(oa, "_history_ids_for_postcommit", lambda *a, **k: (list(history), None))
    state = _postcommit_state(tokenizer=VocabTokenizer(VOCAB))
    result = _store_generation_final_history_snapshot(
        state,
        session_id="session-1",
        prompt_ids=prompt_ids,
        generated={"tokens": generated, "_final_state": _final_state(generated)},
        messages=[ChatMessage(role="user", content="hi")],
        assistant_content='"Nothing',
        thinking_enabled=False,
        policy_fingerprint="policy",
    )
    assert result["stored"] is True
    assert result["mode"] == "generation_final_prefix"
    assert result["reason"] == "generation_boundary_prefix_of_history_after_splice"
    assert result["history_suffix_tokens"] == 1
    assert state.sessions.bank.puts[0]["token_ids"] == final


def test_generation_final_snapshot_accepts_a_length_cut_turn_ending_in_whitespace(monkeypatch):
    # The dump from the 27B probe: the committed turn ends in a newline token
    # the response stripped, so the re-rendered history puts the end-of-turn
    # marker where the newline was. The snapshot must still be an O(1) prefix
    # of the history, not a refused mismatch.
    prompt_ids = [1, 2, 3]
    generated = [WORD, COLON, NL]
    history = [1, 2, 3, WORD, COLON, IM_END, NL]
    monkeypatch.setattr(oa, "_history_ids_for_postcommit", lambda *a, **k: (list(history), None))
    state = _postcommit_state(tokenizer=VocabTokenizer(WS_VOCAB))
    result = _store_generation_final_history_snapshot(
        state,
        session_id="session-1",
        prompt_ids=prompt_ids,
        generated={"tokens": generated, "_final_state": _final_state(generated)},
        messages=[ChatMessage(role="user", content="hi")],
        assistant_content="word:",
        thinking_enabled=False,
        policy_fingerprint="policy",
    )
    assert result["stored"] is True
    assert result["mode"] == "generation_final_prefix"
    assert result["reason"] == "generation_boundary_prefix_of_history_after_splice"
    assert result["token_splice"]["whitespace_tokens"] == 1
    assert state.sessions.bank.puts[0]["token_ids"] == prompt_ids + generated


def test_generation_final_snapshot_still_refuses_a_rewritten_turn(monkeypatch):
    prompt_ids = [1, 2, 3]
    generated = [4, QUOTE_NOTHING, 5, 6]
    history = [1, 2, 3, 4, QUOTE, 12, 5, 6, 99]  # 12 decodes differently: a real rewrite
    monkeypatch.setattr(oa, "_history_ids_for_postcommit", lambda *a, **k: (list(history), None))
    state = _postcommit_state(tokenizer=VocabTokenizer({**VOCAB, 12: "Nothing!"}))
    result = _store_generation_final_history_snapshot(
        state,
        session_id="session-1",
        prompt_ids=prompt_ids,
        generated={"tokens": generated, "_final_state": _final_state(generated)},
        messages=[ChatMessage(role="user", content="hi")],
        assistant_content='"Nothing!',
        thinking_enabled=False,
        policy_fingerprint="policy",
    )
    assert result["stored"] is False
    assert result["mode"] == "unsafe"
