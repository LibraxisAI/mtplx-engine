"""In-forward boundary capture on the Flash-Next family (tiny model, CPU).

A restore boundary at position p needs the recurrent state after token p - 1.
The layers can record it inside one wide forward (the gated-delta recurrence
split at p, the conv tails and the n-gram context sliced), so the prefill loop
does not have to end a forward there. These tests pin the three properties
that make that legal: the forward itself does not change, what is captured at
p equals what a forward ending at p leaves in the cache, and a cache rebuilt
from the capture continues exactly like the uninterrupted forward.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.models.qwen4_exp import (
    ArraysCache,
    TextArgs,
    TextModel,
    boundary_capture_scope,
    take_boundary_captures,
)


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=2,
        hc_lowrank=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ple_layer_ids=[2],
        ngram_vocab_size_base=512,
        heads_per_ngram=2,
        ple_embed_dim=64,
    )


@pytest.fixture()
def tm():
    import mlx_lm.models.cache as cache_module
    import mtplx.models.qwen4_exp as qwen4_exp

    prev = mx.default_device()
    previous_arrays_cache = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    model = TextModel(_tiny_args())
    mx.eval(model.parameters())
    yield model
    qwen4_exp.ArraysCache = previous_arrays_cache
    mx.set_default_device(prev)


def _ids(tokens: int, seed: int = 3) -> mx.array:
    mx.random.seed(seed)
    return mx.random.randint(0, 128, (1, tokens))


def _is_recurrent(entry) -> bool:
    # By ancestry, not by name: in a full run the runtime's patch has swapped
    # the stock class for its FixedArraysCache subclass.
    return any(base.__name__ == "ArraysCache" for base in type(entry).__mro__)


def _recurrent_states(cache):
    return [
        [np.array(leaf) for leaf in entry.state]
        for entry in cache
        if _is_recurrent(entry)
    ]


def _same(a, b, exact=True):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            assert x.shape == y.shape and x.dtype == y.dtype
            if exact:
                assert np.array_equal(x, y)
            else:
                assert np.allclose(x, y, rtol=1e-5, atol=1e-6)


def test_capturing_does_not_change_the_forward(tm):
    ids = _ids(24)
    plain_cache = tm.make_cache()
    plain = tm.model(ids, plain_cache)
    mx.eval(plain)

    cache = tm.make_cache()
    with boundary_capture_scope([7, 16]):
        captured_run = tm.model(ids, cache)
    mx.eval(captured_run)
    assert np.array_equal(np.array(plain), np.array(captured_run))
    _same(_recurrent_states(plain_cache), _recurrent_states(cache))
    captures = take_boundary_captures(cache)
    assert sorted(captures) == [7, 16]
    # Popped: a second take returns nothing, and no forward state leaks into
    # the next one.
    assert take_boundary_captures(cache) == {}


def test_nothing_is_captured_without_the_scope_or_out_of_range(tm):
    ids = _ids(12)
    cache = tm.make_cache()
    tm.model(ids, cache)
    assert take_boundary_captures(cache) == {}
    cache = tm.make_cache()
    with boundary_capture_scope([0, 12, 40, -3]):
        tm.model(ids, cache)
    assert take_boundary_captures(cache) == {}


@pytest.mark.parametrize("edge", [1, 5, 9, 23])
def test_a_capture_equals_the_cache_of_a_forward_that_ends_there(tm, edge):
    ids = _ids(24, seed=edge)
    reference = tm.make_cache()
    mx.eval(tm.model(ids[:, :edge], reference))

    cache = tm.make_cache()
    with boundary_capture_scope([edge]):
        mx.eval(tm.model(ids, cache))
    captured = take_boundary_captures(cache)[edge]
    assert len(captured) == len(cache)
    layout = [
        state for entry, state in zip(cache, captured) if _is_recurrent(entry)
    ]
    # Entries that can be trimmed (the attention caches) carry no capture.
    for entry, state in zip(cache, captured):
        assert (state is None) == (not _is_recurrent(entry))
    _same(_recurrent_states(reference), [[np.array(leaf) for leaf in state] for state in layout], exact=False)


def test_a_capture_taken_in_a_second_chunk_uses_the_carried_state(tm):
    # The boundary sits inside the SECOND forward, so the conv tails and the
    # n-gram context reach back into what the first forward left behind.
    ids = _ids(30, seed=11)
    reference = tm.make_cache()
    mx.eval(tm.model(ids[:, :14], reference))
    mx.eval(tm.model(ids[:, 14:16], reference))

    cache = tm.make_cache()
    mx.eval(tm.model(ids[:, :14], cache))
    with boundary_capture_scope([2]):
        mx.eval(tm.model(ids[:, 14:], cache))
    captured = take_boundary_captures(cache)[2]
    layout = [state for state in captured if state is not None]
    _same(_recurrent_states(reference), [[np.array(leaf) for leaf in state] for state in layout], exact=False)


def test_a_cache_rebuilt_from_the_capture_continues_like_the_uninterrupted_forward(tm):
    ids = _ids(28, seed=5)
    edge = 17
    whole_cache = tm.make_cache()
    with boundary_capture_scope([edge]):
        whole = tm.model(ids, whole_cache)
    mx.eval(whole)
    captured = take_boundary_captures(whole_cache)[edge]

    # A restore: attention entries come from a prefix forward (they are
    # position-indexed), recurrent entries from the capture.
    restored = tm.make_cache()
    mx.eval(tm.model(ids[:, :edge], restored))
    for entry, state in zip(restored, captured):
        if state is not None:
            entry.state = [mx.array(np.array(leaf)) for leaf in state]
    tail = tm.model(ids[:, edge:], restored)
    mx.eval(tail)
    assert np.allclose(np.array(whole)[:, edge:], np.array(tail), rtol=1e-5, atol=1e-6)
    _same(_recurrent_states(whole_cache), _recurrent_states(restored), exact=False)
