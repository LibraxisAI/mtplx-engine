"""The banked generation-final state must cover every committed token.

A scripted tiny target (argmax for stream index i is STOP at i == stop_index,
else 1) drives the real generate_mtpk / generate_ar loops. The draft head is
either "always 1" (so the stop can only arrive as a round-head primary: fresh
sample, deferred correction, or bonus) or omniscient (so it arrives as an
accepted draft inside the verify window). The final trunk-cache offset must
equal len(prompt) + len(tokens): that is exactly the key the server banks the
state under (openai.py commit_final_state_to_bank). Before 2026-09-08 the
round-head exit left the ending primary un-forwarded, so greedy deferred
corrections (and every max_tokens exit) banked a cache one token short.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.generation import _cache_offset, generate_ar, generate_mtpk
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig

VOCAB = 4
STOP = 3


def _onehot(tok: int) -> mx.array:
    row = [0.0] * VOCAB
    row[tok] = 1.0
    return mx.array(row, dtype=mx.float32)


class _OffsetCache:
    def __init__(self) -> None:
        self.offset = 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(int(self.offset), int(n))
        self.offset -= n
        return n


class _TinyTokenizer:
    def decode(self, tokens, **_kw):
        return "".join(str(int(t)) for t in tokens)


class _ScriptedModel:
    def __init__(self, stop_index: int, omniscient_draft: bool) -> None:
        self.stop_index = int(stop_index)
        self.omniscient_draft = bool(omniscient_draft)
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def f(self, i: int) -> int:
        return STOP if i == self.stop_index else 1

    def make_cache(self):
        return [_OffsetCache()]

    def make_mtp_cache(self):
        return [_OffsetCache()]

    def __call__(self, input_ids, *, cache=None, return_hidden=False, hidden_variant=None,
                 emit_logits=True, logits_keep=None):
        length = int(input_ids.shape[1])
        offset = int(cache[0].offset) if cache else 0
        rows = [_onehot(self.f(offset + j + 1)) for j in range(length)]
        if cache:
            cache[0].offset += length
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = mx.stack(rows)[None][:, -keep:, :]
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden_states, next_token_ids, *, mtp_cache=None, concat_order=None,
                    return_hidden=False, mtp_hidden_variant=None, position_offset=None):
        length = int(next_token_ids.shape[1])
        offset = int(mtp_cache[0].offset) if mtp_cache else 0
        if self.omniscient_draft:
            rows = [_onehot(self.f(offset + j + 2)) for j in range(length)]
        else:
            rows = [_onehot(1) for _ in range(length)]
        if mtp_cache:
            mtp_cache[0].offset += length
        logits = mx.stack(rows)[None]
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        return (logits, hidden) if return_hidden else logits

    def mtp_update_cache(self, hidden_states, next_token_ids, *, mtp_cache=None,
                         concat_order=None, position_offset=None):
        if mtp_cache:
            mtp_cache[0].offset += int(next_token_ids.shape[1])
        return hidden_states


def _runtime(model) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=_TinyTokenizer(),
        model_path=Path("tiny"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _gap(prompt, out) -> tuple[int, str | None, bool]:
    fs = out.final_state
    assert fs is not None, "final state must be captured"
    key_len = len(prompt) + len(out.tokens)
    cache_len = _cache_offset(fs.final_trunk_cache)
    origin = out.stats.to_dict().get("finish_stop_origin")
    return key_len - cache_len, origin, bool(fs.safe_to_commit)


PROMPT = [0, 0, 0, 0, 0]
GREEDY = SamplerConfig(temperature=0.0)
COMMON = dict(
    mtp_history_policy="committed",
    verify_strategy="batched",
    stop_token_ids={STOP},
    capture_final_state=True,
)


@pytest.mark.parametrize(
    "label, stop_index, depth, omniscient, sampler, max_tokens",
    [
        ("depth2 stop via bonus primary", len(PROMPT) + 3, 2, False, GREEDY, 16),
        ("depth3 stop via deferred correction", len(PROMPT) + 3, 3, False, GREEDY, 16),
        ("depth3 stop as accepted draft", len(PROMPT) + 3, 3, True, GREEDY, 16),
        ("depth3 sampled stop via correction", len(PROMPT) + 3, 3, False, SamplerConfig(temperature=0.7, top_k=1), 16),
        ("depth3 stop as the very first primary", len(PROMPT) + 1, 3, False, GREEDY, 16),
        ("depth1 stop via fresh primary", len(PROMPT) + 2, 1, False, GREEDY, 16),
        ("depth3 max_tokens exit, no stop", 10_000, 3, False, GREEDY, 7),
        ("depth2 max_tokens exit, no stop", 10_000, 2, False, GREEDY, 5),
    ],
)
def test_mtp_final_state_covers_every_committed_token(label, stop_index, depth, omniscient, sampler, max_tokens):
    out = generate_mtpk(
        _runtime(_ScriptedModel(stop_index, omniscient)),
        PROMPT,
        speculative_depth=depth,
        max_tokens=max_tokens,
        sampler=sampler,
        **COMMON,
    )
    gap, origin, safe = _gap(PROMPT, out)
    assert safe, label
    assert gap == 0, (label, origin, out.tokens, gap)
    if stop_index < 1000:
        assert out.tokens[-1] == STOP, (label, out.tokens)
    mtp_cache = out.final_state.final_committed_mtp_cache
    if mtp_cache is not None:
        # committed history holds one row per token after the first prompt token
        assert int(mtp_cache[0].offset) == len(PROMPT) + len(out.tokens) - 1, label


def test_ar_final_state_control():
    out = generate_ar(
        _runtime(_ScriptedModel(len(PROMPT) + 3, False)),
        PROMPT,
        max_tokens=16,
        sampler=GREEDY,
        stop_token_ids={STOP},
        capture_final_state=True,
    )
    gap, _origin, safe = _gap(PROMPT, out)
    assert safe and gap == 0
