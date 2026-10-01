"""Mid-loop evals in a wide prefill forward (2026-09-20).

One eval at the end of a forward puts every root at the end of MLX's tape, so
the 3-row conv tail each recurrent layer stores (a lazy slice of its
``[rows + 3, 10,240]`` pre-conv stream) is materialized last and every stream
stays alive to the end of the forward: 3.1 GB at 4,096 rows on Flash-Next.
Naming the states in the chunk's eval freed them BETWEEN chunks (3.0 GB off
active memory, measured) and took nothing off the peak (128K cold, 4,096-row
chunks: 100.8 GB against a 100.0 GB line).  Handing the stream and the
finished layers' states to the GPU every four layers lets each stream die when
its layer is done.  On a 300-layer stand-in with 2.1 MB streams the peak over
base fell from 928 MB to 104 MB with identical bits.

Pinned here: the forwards that take it (prefill phase, 1,024 rows or more),
the switch, what is named at each flush (the stream, the recurrent states, any
in-forward boundary captures), and that not one output bit moves: logits,
hidden rows, every cache leaf, through a bare forward, a continued forward and
the real cold prefill loop, on the CPU in float32 and on Metal in bfloat16.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mtplx.models.qwen4_exp as qwen4_exp
from mtplx import demotions
from mtplx.attention_context import attention_phase
from mtplx.generation import _prefill_committed_mtp_history_streaming
from mtplx.models.qwen4_exp import TextModel
from tests.test_prefill_inforward_boundary_capture import (  # noqa: F401  (tiny is a fixture)
    _TinyRuntime,
    _assert_same_leaves,
    _cache_leaves,
    _prompt,
    _tiny_args,
    tiny,
)

SWITCH = "MTPLX_QWEN4_PREFILL_MIDLOOP_EVAL"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in (
        SWITCH,
        "MTPLX_GDN_BOUNDARY_INFORWARD",
        "MTPLX_PREFILL_EVAL_RECURRENT_STATE",
        "MTPLX_PREFILL_CHUNK_TRACE",
    ):
        monkeypatch.delenv(name, raising=False)
    demotions.reset()
    yield
    demotions.reset()


def test_only_wide_prefill_forwards_take_it(monkeypatch):
    assert qwen4_exp._PREFILL_MIDLOOP_MIN_ROWS == 1024
    # Decode, verify and anything outside a prefill loop: never.
    assert qwen4_exp._prefill_midloop_eval_layers(4096) == 0
    with attention_phase("decode"):
        assert qwen4_exp._prefill_midloop_eval_layers(4096) == 0
    with attention_phase("prefill"):
        assert qwen4_exp._prefill_midloop_eval_layers(4096) == 4
        assert qwen4_exp._prefill_midloop_eval_layers(1024) == 4
        assert qwen4_exp._prefill_midloop_eval_layers(1023) == 0
        assert qwen4_exp._prefill_midloop_eval_layers(64) == 0


def test_the_switch(monkeypatch):
    with attention_phase("prefill"):
        for raw, layers in (("0", 0), ("off", 0), ("1", 1), ("8", 8), ("junk", 4), ("", 4), ("-3", 0)):
            monkeypatch.setenv(SWITCH, raw)
            assert qwen4_exp._prefill_midloop_eval_layers(4096) == layers, raw


def _forward(model, tokens, cache, *, layers, monkeypatch):
    monkeypatch.setattr(qwen4_exp, "_PREFILL_MIDLOOP_MIN_ROWS", 8)
    monkeypatch.setenv(SWITCH, str(layers))
    with attention_phase("prefill"):
        logits, hidden = model(
            mx.array([tokens]), cache=cache, return_hidden=True, emit_logits=True
        )
    mx.eval(logits, hidden, *[leaf for entry in cache for leaf in _state_arrays(entry)])
    return np.array(logits.astype(mx.float32)), np.array(hidden.astype(mx.float32))


def _state_arrays(entry):
    state = getattr(entry, "state", None)
    return [leaf for leaf in (state or []) if isinstance(leaf, mx.array)]


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_not_one_bit_moves_in_a_forward_or_in_the_one_after_it(tiny, monkeypatch, layers):
    prompt = _prompt(96)
    reference_cache = tiny.make_cache()
    ref_first = _forward(tiny, prompt[:60], reference_cache, layers=0, monkeypatch=monkeypatch)
    ref_second = _forward(tiny, prompt[60:], reference_cache, layers=0, monkeypatch=monkeypatch)
    cache = tiny.make_cache()
    first = _forward(tiny, prompt[:60], cache, layers=layers, monkeypatch=monkeypatch)
    second = _forward(tiny, prompt[60:], cache, layers=layers, monkeypatch=monkeypatch)
    for got, want in zip(first + second, ref_first + ref_second):
        assert got.shape == want.shape and np.array_equal(got, want, equal_nan=True)
    _assert_same_leaves(_cache_leaves(cache), _cache_leaves(reference_cache))


def test_each_flush_names_the_stream_and_the_finished_layers_states(tiny, monkeypatch):
    calls: list[list[mx.array]] = []
    real = mx.async_eval

    def spy(*arrays):
        calls.append(list(arrays))
        return real(*arrays)

    monkeypatch.setattr(qwen4_exp.mx, "async_eval", spy)
    cache = tiny.make_cache()
    _forward(tiny, _prompt(40), cache, layers=1, monkeypatch=monkeypatch)
    # Four layers, a flush after each but the last (the loop's own eval ends
    # the forward).
    assert len(calls) == 3
    for index, arrays in enumerate(calls):
        entry = cache[index]
        assert isinstance(entry, qwen4_exp.ArraysCache)
        named = {id(a) for a in arrays}
        assert len(arrays) == 1 + len(_state_arrays(entry))
        for leaf in _state_arrays(entry):
            assert id(leaf) in named
        assert arrays[0].shape[1] == 40  # the widened stream comes first
    # The PLE layer keeps four states (conv tail, delta state, PLE conv tail,
    # n-gram context); all four are named.
    assert max(len(arrays) for arrays in calls) == 5
    # Two layers a flush: one flush (after layer 1), naming both layers.
    calls.clear()
    cache = tiny.make_cache()
    _forward(tiny, _prompt(40), cache, layers=2, monkeypatch=monkeypatch)
    assert len(calls) == 1
    assert len(calls[0]) == 1 + len(_state_arrays(cache[0])) + len(_state_arrays(cache[1]))


def test_a_flush_names_the_in_forward_boundary_captures(tiny, monkeypatch):
    calls: list[int] = []
    real = mx.async_eval

    def spy(*arrays):
        calls.append(len(arrays))
        return real(*arrays)

    monkeypatch.setattr(qwen4_exp.mx, "async_eval", spy)
    cache = tiny.make_cache()
    _forward(tiny, _prompt(40), cache, layers=1, monkeypatch=monkeypatch)
    plain = list(calls)
    calls.clear()
    cache = tiny.make_cache()
    monkeypatch.setattr(qwen4_exp, "_PREFILL_MIDLOOP_MIN_ROWS", 8)
    monkeypatch.setenv(SWITCH, "1")
    with attention_phase("prefill"), qwen4_exp.boundary_capture_scope((16, 32)):
        logits = tiny(mx.array([_prompt(40)]), cache=cache)
    mx.eval(logits)
    assert len(calls) == len(plain)
    assert all(armed > bare for armed, bare in zip(calls, plain))
    captured = qwen4_exp.take_boundary_captures(cache)
    assert sorted(captured) == [16, 32]


def test_nothing_is_flushed_without_the_prefill_phase_or_below_the_width(tiny, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(qwen4_exp.mx, "async_eval", lambda *arrays: calls.append(len(arrays)))
    monkeypatch.setenv(SWITCH, "1")
    cache = tiny.make_cache()
    with attention_phase("prefill"):
        mx.eval(tiny(mx.array([_prompt(40)]), cache=cache))  # 40 rows < 1,024
    mx.eval(tiny(mx.array([_prompt(8)]), cache=cache))  # no phase
    monkeypatch.setattr(qwen4_exp, "_PREFILL_MIDLOOP_MIN_ROWS", 8)
    mx.eval(tiny(mx.array([_prompt(16)]), cache=cache))  # wide enough, no phase
    with attention_phase("decode"):
        mx.eval(tiny(mx.array([_prompt(16)]), cache=cache))
    assert calls == []


def _cold(model, prompt, *, layers, monkeypatch):
    monkeypatch.setattr(qwen4_exp, "_PREFILL_MIDLOOP_MIN_ROWS", 8)
    monkeypatch.setenv(SWITCH, str(layers))
    rt = _TinyRuntime(model)
    sink: list = []
    out = _prefill_committed_mtp_history_streaming(rt, list(prompt), gdn_boundary_sink=sink)
    cache, logits, hidden = out[0], out[1], out[2]
    mx.eval(logits, hidden)
    return rt, cache, logits, hidden, sink


@pytest.mark.parametrize("inforward", [False, True])
def test_the_real_cold_loop_is_bit_identical_with_and_without_it(tiny, monkeypatch, inforward):
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_INFORWARD", "1" if inforward else "0")
    prompt = _prompt(131)
    ref_rt, ref_cache, ref_logits, ref_hidden, ref_sink = _cold(
        tiny, prompt, layers=0, monkeypatch=monkeypatch
    )
    rt, cache, logits, hidden, sink = _cold(tiny, prompt, layers=1, monkeypatch=monkeypatch)
    assert rt.forwards == ref_rt.forwards
    assert np.array_equal(np.array(logits), np.array(ref_logits), equal_nan=True)
    assert np.array_equal(np.array(hidden), np.array(ref_hidden), equal_nan=True)
    _assert_same_leaves(_cache_leaves(cache), _cache_leaves(ref_cache))
    assert [record[0] for record in sink] == [record[0] for record in ref_sink]
    for got, want in zip(sink, ref_sink):
        _assert_same_leaves(
            [leaf for leaf in _snapshot_leaves(got[1])],
            [leaf for leaf in _snapshot_leaves(want[1])],
        )
    for got, want in zip(rt.history_rows, ref_rt.history_rows):
        assert np.array_equal(got, want, equal_nan=True)


def _snapshot_leaves(snapshot):
    from tests.test_prefill_inforward_boundary_capture import _leaves

    return [_leaves(state) for state in snapshot.states]


needs_metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="bfloat16 forwards run on Metal"
)


@needs_metal
def test_not_one_bit_moves_on_metal_in_bfloat16(monkeypatch):
    import mlx_lm.models.cache as cache_module
    from mlx.utils import tree_map

    previous = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    try:
        mx.random.seed(3)
        model = TextModel(_tiny_args())
        model.update(tree_map(lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p, model.parameters()))
        mx.eval(model.parameters())
        prompt = _prompt(200, seed=11)
        outs = {}
        for layers in (0, 1, 2):
            cache = model.make_cache()
            first = _forward(model, prompt[:120], cache, layers=layers, monkeypatch=monkeypatch)
            second = _forward(model, prompt[120:], cache, layers=layers, monkeypatch=monkeypatch)
            outs[layers] = (first + second, _cache_leaves(cache))
        for layers in (1, 2):
            for got, want in zip(outs[layers][0], outs[0][0]):
                assert np.array_equal(got, want, equal_nan=True)
            _assert_same_leaves(outs[layers][1], outs[0][1])
    finally:
        qwen4_exp.ArraysCache = previous
