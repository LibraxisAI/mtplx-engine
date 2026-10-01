"""Non-finite logits are a truthful failure, never token 0; and the
pipelined-AR device sampler follows the reference nucleus law."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.generation import _mx_lazy_sample, _mx_lazy_shape, _sample_from_logits
from mtplx.sampling import NonFiniteLogitsError, SamplerConfig, distribution_from_logits, softmax


def test_softmax_names_the_fault():
    with pytest.raises(NonFiniteLogitsError) as info:
        softmax(np.array([1.0, np.nan, np.inf]), temperature=1.0)
    assert info.value.nan_count == 1 and info.value.inf_count == 1
    assert "softmax" in str(info.value)


@pytest.mark.parametrize(
    "row",
    [
        mx.full((64,), float("nan"), dtype=mx.float32),
        mx.array([float("inf")] + [0.0] * 63, dtype=mx.float32),
        mx.array([float("nan")] + [0.0] * 63, dtype=mx.float32),
        mx.full((64,), float("-inf"), dtype=mx.float32),
    ],
)
@pytest.mark.parametrize("temperature", [0.0, 0.6, 1.0])
def test_sample_from_logits_raises_on_every_non_finite_shape(row, temperature):
    config = SamplerConfig(temperature=temperature, top_p=0.95, top_k=20)
    with pytest.raises(NonFiniteLogitsError):
        _sample_from_logits(row, config, np.random.default_rng(0))


def test_greedy_finite_row_still_samples():
    row = mx.array([0.0, 3.0, 1.0], dtype=mx.float32)
    token, _ = _sample_from_logits(row, SamplerConfig(temperature=0.0), np.random.default_rng(0))
    assert token == 1


@pytest.mark.parametrize("temperature", [0.6, 1.0])
@pytest.mark.parametrize("sharpness", [0.3, 1.0, 2.0, 4.0])
def test_lazy_shape_matches_the_reference_law(temperature, sharpness):
    """The device nucleus (full-vocabulary softmax) equals sampling.apply_top_p_top_k."""

    rng = np.random.default_rng(int(sharpness * 10) + int(temperature * 10))
    config = SamplerConfig(temperature=temperature, top_p=0.95, top_k=20)
    vocab = 4096
    worst = 0.0
    for _ in range(12):
        # fp32 rows with distinct values: no top-k boundary ties, so the two
        # tie rules (device order vs lowest id) cannot differ here
        logits = rng.normal(size=vocab).astype(np.float32) * sharpness + 20.0
        row = mx.array(logits)
        ids, log_weights, bad = _mx_lazy_shape(row, config)
        mx.eval(ids, log_weights, bad)
        assert not bool(bad.item())
        probs = np.zeros(vocab)
        lw = np.asarray(log_weights, dtype=np.float64)
        keep = np.isfinite(lw)
        p = np.exp(lw[keep] - lw[keep].max())
        probs[np.asarray(ids)[keep]] = p / p.sum()
        reference = distribution_from_logits(logits.astype(np.float64), config)
        tv = 0.5 * float(np.abs(probs - reference).sum())
        worst = max(worst, tv)
        assert set(np.flatnonzero(probs > 0)) == set(np.flatnonzero(reference > 0))
    assert worst < 1e-5, worst


def test_lazy_sample_returns_the_sentinel_on_a_non_finite_row():
    config = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
    key = mx.random.key(1)
    for row in (
        mx.full((256,), float("nan"), dtype=mx.float32),
        mx.array([float("inf")] + [0.0] * 255, dtype=mx.float32),
        mx.full((256,), float("-inf"), dtype=mx.float32),
    ):
        token = _mx_lazy_sample(row, config, key)
        mx.eval(token)
        assert int(token.item()) == -1


def test_lazy_sample_draws_a_finite_token_from_the_support():
    config = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
    rng = np.random.default_rng(3)
    logits = rng.normal(size=1024).astype(np.float32) * 2.0
    row = mx.array(logits)
    reference = distribution_from_logits(logits.astype(np.float64), config)
    support = set(np.flatnonzero(reference > 0).tolist())
    key = mx.random.key(7)
    for _ in range(64):
        key, sub = mx.random.split(key)
        token = _mx_lazy_sample(row, config, sub)
        mx.eval(token)
        assert int(token.item()) in support


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_masked_minus_inf_rows_are_legitimate(temperature):
    """Grammar masks and penalties put -inf on illegal tokens: not a fault."""

    row = mx.array([float("-inf"), 2.0, float("-inf"), 1.0], dtype=mx.float32)
    config = SamplerConfig(temperature=temperature, top_p=0.95, top_k=20)
    token, _ = _sample_from_logits(row, config, np.random.default_rng(0))
    assert token in (1, 3)
    if temperature == 0.0:
        assert token == 1
    ids, log_weights, bad = _mx_lazy_shape(mx.concatenate([row, mx.zeros((28,), dtype=mx.float32)]), SamplerConfig(temperature=1.0, top_p=0.95, top_k=4))
    mx.eval(bad)
    assert not bool(bad.item())
