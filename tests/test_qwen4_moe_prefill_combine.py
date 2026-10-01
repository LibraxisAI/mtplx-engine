"""Flash-Next's prefill MoE combine as one kernel (2026-09-23).

``mtplx.kernels.qwen4_moe_prefill_combine`` replaces the stock
unsort -> weight -> sum -> + shared tail of a prefill-width MoE forward.  It
must be bit-identical to that tail, at the model's own geometry and at the
chunk widths prefill runs, and the block must keep the stock forward for
decode/verify widths and with ``MTPLX_QWEN4_MOE_PREFILL_COMBINE=0``.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, _gather_sort, _scatter_unsort

from mtplx.kernels.qwen4_moe_prefill_combine import (
    combine_eligible,
    moe_prefill_combine,
    moe_prefill_combine_reference,
)
from mtplx.models import qwen4_exp


def _bits(a: mx.array) -> np.ndarray:
    mx.eval(a)
    return np.array(a.view(mx.uint16))


def _stock_inputs(rows, top_k, hidden, scale, seed=0, experts=512):
    mx.random.seed(seed)
    inds = mx.random.randint(0, experts, (1, rows, top_k)).astype(mx.uint32)
    _, _, inv = _gather_sort(mx.zeros((1, rows, 1, 1, 1), dtype=mx.bfloat16), inds)
    y_sorted = (mx.random.normal((rows * top_k, 1, hidden)) * scale).astype(mx.bfloat16)
    gates = mx.softmax((mx.random.normal((1, rows, top_k)) * 3).astype(mx.bfloat16), axis=-1, precise=True)
    scores = gates / gates.sum(axis=-1, keepdims=True)
    shared = (mx.random.normal((1, rows, hidden)) * scale).astype(mx.bfloat16)
    return inds, inv, y_sorted, scores, shared


@pytest.mark.parametrize(
    "rows,top_k,hidden,scale",
    [(4096, 10, 2560, 1.0),   # the Flash-Next chunk
     (2048, 10, 2560, 30.0),
     (300, 10, 2560, 0.01),
     (128, 8, 512, 1.0),
     (97, 3, 256, 5.0)],
)
def test_the_kernel_is_bit_identical_to_the_stock_tail(rows, top_k, hidden, scale):
    inds, inv, y_sorted, scores, shared = _stock_inputs(rows, top_k, hidden, scale, seed=rows)
    # The stock tail exactly as SwitchGLU + Qwen3NextSparseMoeBlock run it.
    y = _scatter_unsort(y_sorted, inv, inds.shape).squeeze(-2)
    stock = ((y * scores[..., None]).sum(axis=-2) + shared).reshape(rows, hidden)
    args = (y_sorted.reshape(rows * top_k, hidden), inv, scores.reshape(rows, top_k), shared.reshape(rows, hidden))
    assert combine_eligible(*args)
    assert np.array_equal(_bits(moe_prefill_combine(*args)), _bits(stock))
    assert np.array_equal(_bits(moe_prefill_combine_reference(*args)), _bits(stock))


def test_ineligible_inputs_take_the_reference():
    inds, inv, y_sorted, scores, shared = _stock_inputs(64, 10, 256, 1.0)
    args = (y_sorted.reshape(640, 256).astype(mx.float32), inv, scores.reshape(64, 10), shared.reshape(64, 256))
    assert not combine_eligible(*args)
    got = moe_prefill_combine(*args)
    assert got.dtype == mx.float32 and got.shape == (64, 256)


def _block(hidden=256, experts=16, top_k=4, inter=64, bits=4, group=32):
    args = SimpleNamespace(
        hidden_size=hidden, moe_intermediate_size=inter, shared_expert_intermediate_size=inter,
        norm_topk_prob=True, num_experts=experts, num_experts_per_tok=top_k,
    )
    block = qwen4_exp.SparseMoeBlock(args)
    mx.random.seed(7)
    gu = (mx.random.normal((experts, 2 * inter, hidden)) * 0.05).astype(mx.bfloat16)
    gu_w, gu_s, gu_b = mx.quantize(gu, group_size=group, bits=bits)
    down = QuantizedSwitchLinear(inter, hidden, experts, bias=False, group_size=group, bits=bits)
    down.set_dtype(mx.bfloat16)
    block.switch_mlp = qwen4_exp._FusedGateUpSwitchGLU(down, gu_w, gu_s, gu_b, group, bits, "affine")
    block.set_dtype(mx.bfloat16)
    return block


def test_the_block_forward_is_bit_identical_with_and_without_the_kernel(monkeypatch):
    import mtplx.kernels.qwen4_moe_prefill_combine as combine

    block = _block()
    x = mx.random.normal((1, 96, 256)).astype(mx.bfloat16)
    calls = []
    real = combine.moe_prefill_combine
    monkeypatch.setattr(combine, "moe_prefill_combine", lambda *a: calls.append(1) or real(*a))
    monkeypatch.delenv("MTPLX_QWEN4_MOE_PREFILL_COMBINE", raising=False)
    fused = block(x)
    monkeypatch.setenv("MTPLX_QWEN4_MOE_PREFILL_COMBINE", "0")
    stock = block(x)
    assert calls == [1]
    assert fused.shape == stock.shape == x.shape
    assert np.array_equal(_bits(fused), _bits(stock))


@pytest.mark.parametrize("rows", [1, 4, 31])
def test_narrow_forwards_keep_the_stock_forward(monkeypatch, rows):
    import mtplx.kernels.qwen4_moe_prefill_combine as combine

    def refuse(*args):  # pragma: no cover - must not be reached
        raise AssertionError("a narrow forward took the prefill combine")

    monkeypatch.setattr(combine, "moe_prefill_combine", refuse)
    monkeypatch.delenv("MTPLX_QWEN4_MOE_PREFILL_COMBINE", raising=False)
    block = _block()
    out = block(mx.random.normal((1, rows, 256)).astype(mx.bfloat16))
    mx.eval(out)
    assert out.shape == (1, rows, 256)


def test_the_switch_turns_the_kernel_off(monkeypatch):
    block = _block()
    x = mx.zeros((1, 64, 256), dtype=mx.bfloat16)
    monkeypatch.delenv("MTPLX_QWEN4_MOE_PREFILL_COMBINE", raising=False)
    assert qwen4_exp._moe_prefill_combine_applies(block, x)
    for off in ("0", "false", "no", "off"):
        monkeypatch.setenv("MTPLX_QWEN4_MOE_PREFILL_COMBINE", off)
        assert not qwen4_exp._moe_prefill_combine_applies(block, x)
    monkeypatch.delenv("MTPLX_QWEN4_MOE_PREFILL_COMBINE", raising=False)
    assert not qwen4_exp._moe_prefill_combine_applies(block, x.astype(mx.float32))
    block.switch_mlp = nn.Identity()
    assert not qwen4_exp._moe_prefill_combine_applies(block, x)
