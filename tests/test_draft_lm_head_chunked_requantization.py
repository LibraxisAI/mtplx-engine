"""Issue #483: the draft LM head is requantized in row chunks (no dense
intermediate), the result is bit-identical to the whole-tensor path, a zeroed
head is refused, and the install falls back to the resident target head."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import mtplx.draft_lm_head as dlh


def _quantized_head(rows: int = 96, cols: int = 256, bits: int = 8, group_size: int = 64) -> nn.QuantizedLinear:
    linear = nn.Linear(cols, rows, bias=False)
    linear.weight = mx.random.normal((rows, cols), dtype=mx.float32).astype(mx.bfloat16)
    head = nn.QuantizedLinear.from_linear(linear, group_size=group_size, bits=bits)
    mx.eval(head.weight, head.scales, head.biases)
    return head


def test_chunked_requantization_is_bit_identical_to_whole_tensor():
    head = _quantized_head()
    chunked = dlh._requantize_in_row_chunks(head, bits=4, group_size=64, mode="affine", rows_per_chunk=16)
    dense = mx.dequantize(head.weight, head.scales, head.biases, group_size=64, bits=8, mode="affine").astype(mx.bfloat16)
    whole = mx.quantize(dense, group_size=64, bits=4, mode="affine")
    mx.eval(chunked.weight, chunked.scales, chunked.biases, *whole)
    assert mx.array_equal(chunked.weight, whole[0]).item()
    assert mx.array_equal(chunked.scales, whole[1]).item()
    assert mx.array_equal(chunked.biases, whole[2]).item()
    assert chunked.bits == 4 and chunked.group_size == 64 and chunked.mode == "affine"
    x = mx.random.normal((3, 256)).astype(mx.bfloat16)
    reference = mx.quantized_matmul(x, whole[0], whole[1], whole[2], group_size=64, bits=4)
    mx.eval(reference)
    assert mx.array_equal(chunked(x), reference).item()


def test_a_zeroed_head_is_refused(monkeypatch):
    head = _quantized_head()
    real_quantize = mx.quantize

    def zeroed(w, group_size, bits, mode="affine"):
        parts = real_quantize(w, group_size=group_size, bits=bits, mode=mode)
        return tuple(mx.zeros_like(p) for p in parts)

    monkeypatch.setattr(mx, "quantize", zeroed)
    with pytest.raises(dlh.ZeroedDraftHeadError):
        dlh._requantize_in_row_chunks(head, bits=4, group_size=64, mode="affine")


def test_install_falls_back_to_the_resident_head_when_requantization_fails(monkeypatch, capsys):
    head = _quantized_head()
    text = SimpleNamespace(lm_head=head, mtp=SimpleNamespace(layers=[]), args=SimpleNamespace(tie_word_embeddings=False))
    rt = SimpleNamespace(model=text)

    def boom(*_a, **_k):
        raise dlh.ZeroedDraftHeadError("device ran out of memory")

    monkeypatch.setattr(dlh, "_make_requantized_head", boom)
    monkeypatch.setenv("MTPLX_FRSPEC_DRAFT", "0")
    report = dlh._install_draft_lm_head(rt, bits=4, group_size=64, mode="affine")
    assert text._mtplx_draft_lm_head is head
    assert report["reused_existing_quantization"] is True
    assert "requantization_failed" in report
    assert "reusing the target" in capsys.readouterr().err


def test_install_uses_the_chunked_path(monkeypatch):
    head = _quantized_head()
    text = SimpleNamespace(lm_head=head, mtp=SimpleNamespace(layers=[]), args=SimpleNamespace(tie_word_embeddings=False))
    rt = SimpleNamespace(model=text)
    monkeypatch.setenv("MTPLX_FRSPEC_DRAFT", "0")
    report = dlh._install_draft_lm_head(rt, bits=4, group_size=64, mode="affine")
    assert report["draft_only"]["bits"] == 4
    assert text._mtplx_draft_lm_head is not head
    assert text._mtplx_draft_lm_head.bits == 4
