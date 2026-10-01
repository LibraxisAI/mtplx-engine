"""Draft-head history appends compute the cache and nothing else (2026-09-20).

Appending prompt history to Flash-Next's draft head ran its whole decoder
layer over every prefill chunk (block selection, attention, the routed
experts, both hyper-connection writes) and the loop then threw the output
away: ``_append_mtp_history`` evaluates it and returns the elapsed time.
0.09 to 0.12 s per 4,096-row chunk in tonight's traces, 3 percent of a cold
prefill at every length.  What the pass leaves behind is the head's QSA cache,
and that is written before any of that work starts.

Pinned here: with the route on, every leaf of the draft head's cache is
bit-identical to the full pass, chunk after chunk, the first draft step after
it gives the same logits and the same deeper-draft state, the experts and the
attention kernel are never reached, and everything outside a prefill loop
(and the rollback switch) still runs the whole layer and returns its output.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mtplx.models.qwen4_exp as qwen4_exp
from mtplx.attention_context import attention_phase
from mtplx.generation import _prefill_committed_mtp_history_streaming
from mtplx.models.qwen4_exp import Qwen4ExpMTP, TextModel
from tests.test_prefill_inforward_boundary_capture import (  # noqa: F401  (tiny is a fixture)
    _TinyRuntime,
    _prompt,
    _tiny_args,
    tiny,
)

SWITCH = "MTPLX_QWEN4_MTP_HISTORY_CACHE_ONLY"
CHUNKS = (37, 64, 5, 2, 48)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in (SWITCH, "MTPLX_LAZY_MTP_HISTORY_APPEND", "MTPLX_PREFILL_CHUNK_TRACE"):
        monkeypatch.delenv(name, raising=False)
    yield


def _with_head(model: TextModel, dtype=None) -> TextModel:
    mx.random.seed(5)
    head = Qwen4ExpMTP(model.args)
    if dtype is not None:
        from mlx.utils import tree_map

        head.update(tree_map(lambda p: p.astype(dtype) if p.dtype == mx.float32 else p, head.parameters()))
    mx.eval(head.parameters())
    model.mtp = head
    return model


def _cache_bits(cache) -> list[np.ndarray]:
    entry = cache[0]
    offset = int(entry.offset)
    leaves = [
        entry.kv.keys[..., :offset, :],
        entry.kv.values[..., :offset, :],
        entry.raw_keys[:, :offset, :],
        entry.pooled[:, : entry.pooled_len, :],
        entry.pooled_f32_t[..., : entry.pooled_len],
    ]
    mx.eval(*leaves)
    out = [np.array(leaf.astype(mx.float32)) for leaf in leaves]
    out.append(np.array([offset, int(entry.pooled_len)]))
    return out


def _append_all(model, *, cache_only: bool, monkeypatch, dtype=mx.float32, phase="prefill"):
    monkeypatch.setenv(SWITCH, "1" if cache_only else "0")
    width = model.args.hidden_size * model.args.hc_count
    cache = model.make_mtp_cache()
    returned = []
    mx.random.seed(21)
    start = 0
    tokens = _prompt(sum(CHUNKS) + 1, seed=3)
    for rows in CHUNKS:
        widened = mx.random.normal((1, rows, width)).astype(dtype)
        ids = mx.array([tokens[start + 1 : start + 1 + rows]])
        with attention_phase(phase):
            out = model.mtp_update_cache(widened, ids, mtp_cache=cache)
        mx.eval(out)
        returned.append(out)
        start += rows
    return cache, returned


def _first_draft(model, cache, dtype=mx.float32):
    width = model.args.hidden_size * model.args.hc_count
    mx.random.seed(99)
    widened = mx.random.normal((1, 1, width)).astype(dtype)
    logits, state = model.mtp_forward(widened, mx.array([[7]]), mtp_cache=cache, return_hidden=True)
    mx.eval(logits, state)
    return np.array(logits.astype(mx.float32)), np.array(state.astype(mx.float32))


def test_every_cache_leaf_and_the_first_draft_are_bit_identical(tiny, monkeypatch):
    model = _with_head(tiny)
    full_cache, full_out = _append_all(model, cache_only=False, monkeypatch=monkeypatch)
    only_cache, only_out = _append_all(model, cache_only=True, monkeypatch=monkeypatch)
    assert all(isinstance(out, mx.array) for out in full_out)
    assert all(isinstance(out, list) and out for out in only_out)
    for got, want in zip(_cache_bits(only_cache), _cache_bits(full_cache)):
        assert got.shape == want.shape and np.array_equal(got, want, equal_nan=True)
    assert int(only_cache[0].offset) == sum(CHUNKS)
    for got, want in zip(_first_draft(model, only_cache), _first_draft(model, full_cache)):
        assert np.array_equal(got, want, equal_nan=True)


def test_the_experts_and_the_attention_kernel_are_never_reached(tiny, monkeypatch):
    model = _with_head(tiny)
    calls = {"moe": 0, "sdpa": 0, "select": 0}
    moe = qwen4_exp.SparseMoeBlock.__call__
    select = qwen4_exp.QSAIndexer._select_eager
    sdpa = qwen4_exp._verify_sdpa

    def counted_moe(self, x):
        calls["moe"] += 1
        return moe(self, x)

    def counted_select(self, *args, **kwargs):
        calls["select"] += 1
        return select(self, *args, **kwargs)

    def counted_sdpa(*args, **kwargs):
        calls["sdpa"] += 1
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(qwen4_exp.SparseMoeBlock, "__call__", counted_moe)
    monkeypatch.setattr(qwen4_exp.QSAIndexer, "_select_eager", counted_select)
    monkeypatch.setattr(qwen4_exp, "_verify_sdpa", counted_sdpa)
    _append_all(model, cache_only=False, monkeypatch=monkeypatch)
    assert calls["moe"] == len(CHUNKS) and calls["sdpa"] == len(CHUNKS) and calls["select"] >= 1
    calls.update(moe=0, sdpa=0, select=0)
    _append_all(model, cache_only=True, monkeypatch=monkeypatch)
    assert calls == {"moe": 0, "sdpa": 0, "select": 0}


@pytest.mark.parametrize("phase", ["ar_decode", "decode", "verify"])
def test_outside_a_prefill_loop_the_whole_layer_runs(tiny, monkeypatch, phase):
    model = _with_head(tiny)
    _cache, returned = _append_all(model, cache_only=True, monkeypatch=monkeypatch, phase=phase)
    width = model.args.hidden_size * model.args.hc_count
    for out, rows in zip(returned, CHUNKS):
        assert isinstance(out, mx.array) and out.shape == (1, rows, width)


def test_the_flag_never_leaks_out_of_the_append(tiny, monkeypatch):
    model = _with_head(tiny)
    _append_all(model, cache_only=True, monkeypatch=monkeypatch)
    assert qwen4_exp._QSA_HISTORY_ONLY.get() is False
    # A trunk forward right after still attends and still selects.
    cache = model.make_cache()
    with attention_phase("prefill"):
        logits = model(mx.array([_prompt(40)]), cache=cache)
    mx.eval(logits)
    assert logits.shape[1] == 40


def test_a_failing_append_resets_the_flag(tiny, monkeypatch):
    model = _with_head(tiny)
    monkeypatch.setenv(SWITCH, "1")

    def boom(self, x, cache):
        raise RuntimeError("attention failed")

    monkeypatch.setattr(qwen4_exp.Attention, "__call__", boom)
    width = model.args.hidden_size * model.args.hc_count
    with attention_phase("prefill"), pytest.raises(RuntimeError, match="attention failed"):
        model.mtp_update_cache(
            mx.zeros((1, 4, width)), mx.array([[1, 2, 3, 4]]), mtp_cache=model.make_mtp_cache()
        )
    assert qwen4_exp._QSA_HISTORY_ONLY.get() is False


class _HeadRuntime(_TinyRuntime):
    """The tiny runtime with the family's real draft-head history route."""

    def make_mtp_cache(self):
        return self.model.make_mtp_cache()

    def update_mtp_cache(self, hidden_states, token_ids, *, mtp_cache=None, **_kwargs):
        return self.model.mtp_update_cache(hidden_states, token_ids, mtp_cache=mtp_cache)


def test_the_real_cold_loop_leaves_the_same_draft_head_cache(tiny, monkeypatch):
    model = _with_head(tiny)
    prompt = _prompt(131)
    caches = {}
    engaged = {False: 0, True: 0}
    collect = qwen4_exp._qsa_cache_arrays
    for cache_only in (False, True):
        monkeypatch.setenv(SWITCH, "1" if cache_only else "0")

        def counted(cache, _arm=cache_only):
            engaged[_arm] += 1
            return collect(cache)

        monkeypatch.setattr(qwen4_exp, "_qsa_cache_arrays", counted)
        rt = _HeadRuntime(model)
        out = _prefill_committed_mtp_history_streaming(rt, list(prompt), gdn_boundary_sink=[])
        mx.eval(out[1], out[2])
        mtp_cache = next(
            item for item in out if isinstance(item, list) and item and isinstance(item[0], qwen4_exp.QSACache)
            and item is not out[0]
        )
        caches[cache_only] = (_cache_bits(mtp_cache), np.array(out[1]), _first_draft(model, mtp_cache))
    # The loop's appends run in the prefill phase, so the route engaged on
    # every chunk of one arm and on none of the other.
    assert engaged[False] == 0 and engaged[True] >= 3
    for got, want in zip(caches[True][0], caches[False][0]):
        assert np.array_equal(got, want, equal_nan=True)
    assert np.array_equal(caches[True][1], caches[False][1], equal_nan=True)
    for got, want in zip(caches[True][2], caches[False][2]):
        assert np.array_equal(got, want, equal_nan=True)


needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="bfloat16 runs on Metal")


@needs_metal
def test_bit_identical_on_metal_in_bfloat16(monkeypatch):
    import mlx_lm.models.cache as cache_module
    from mlx.utils import tree_map

    previous = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    try:
        mx.random.seed(1)
        model = TextModel(_tiny_args())
        model.update(tree_map(lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p, model.parameters()))
        mx.eval(model.parameters())
        _with_head(model, dtype=mx.bfloat16)
        full_cache, _ = _append_all(model, cache_only=False, monkeypatch=monkeypatch, dtype=mx.bfloat16)
        only_cache, _ = _append_all(model, cache_only=True, monkeypatch=monkeypatch, dtype=mx.bfloat16)
        for got, want in zip(_cache_bits(only_cache), _cache_bits(full_cache)):
            assert np.array_equal(got, want, equal_nan=True)
        for got, want in zip(
            _first_draft(model, only_cache, mx.bfloat16), _first_draft(model, full_cache, mx.bfloat16)
        ):
            assert np.array_equal(got, want, equal_nan=True)
    finally:
        qwen4_exp.ArraysCache = previous
