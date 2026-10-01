from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mtplx.draft_lm_head import _install_draft_lm_head
from mtplx.mtp_adapters import LoRALinear


class _TiedEmbeddingText(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        embedding = nn.Embedding(32, 32)
        embedding.weight = mx.ones((32, 32), dtype=mx.bfloat16)
        self.model = SimpleNamespace(
            embed_tokens=nn.QuantizedEmbedding.from_embedding(
                embedding,
                group_size=32,
                bits=4,
                mode="affine",
            )
        )
        self.args = SimpleNamespace(tie_word_embeddings=True)


class _StepMTPText(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared_head = nn.Linear(32, 64, bias=False)
        shared_head.weight = mx.ones((64, 32), dtype=mx.bfloat16)
        self.mtp = SimpleNamespace(
            layers=[
                SimpleNamespace(shared_head_head=shared_head),
            ]
        )
        self.lm_head = nn.Linear(32, 64, bias=False)


def test_install_draft_lm_head_supports_tied_quantized_embeddings() -> None:
    rt = SimpleNamespace(model=_TiedEmbeddingText())

    report = _install_draft_lm_head(rt, bits=4, group_size=32, mode="affine")
    head = rt.model._mtplx_draft_lm_head
    logits = head(mx.ones((1, 2, 32), dtype=mx.bfloat16))
    mx.eval(logits)

    assert report["source"] == "tied_embedding"
    assert report["reused_existing_quantization"] is True
    assert logits.shape == (1, 2, 32)


def test_install_draft_lm_head_quantizes_dense_linear_head() -> None:
    linear = nn.Linear(32, 64, bias=False)
    linear.weight = mx.ones((64, 32), dtype=mx.bfloat16)
    rt = SimpleNamespace(model=SimpleNamespace(lm_head=linear))

    report = _install_draft_lm_head(rt, bits=4, group_size=32, mode="affine")
    head = rt.model._mtplx_draft_lm_head
    logits = head(mx.ones((1, 2, 32), dtype=mx.bfloat16))
    mx.eval(logits)

    assert report["source"] == "dense_lm_head"
    assert report["reused_existing_quantization"] is False
    assert logits.shape == (1, 2, 64)


def test_install_draft_lm_head_quantizes_step_mtp_shared_heads() -> None:
    rt = SimpleNamespace(model=_StepMTPText())

    report = _install_draft_lm_head(rt, bits=3, group_size=32, mode="affine")
    head = rt.model.mtp.layers[0].shared_head_head
    logits = head(mx.ones((1, 2, 32), dtype=mx.bfloat16))
    mx.eval(logits)

    assert report["source"] == "step_mtp_shared_head"
    assert report["layers"][0]["layer"] == 0
    assert report["layers"][0]["draft_only"]["bits"] == 3
    assert head.bits == 3
    assert rt.model.lm_head.__class__ is nn.Linear
    assert logits.shape == (1, 2, 64)


def test_install_draft_lm_head_quantizes_lora_wrapped_step_shared_heads() -> None:
    rt = SimpleNamespace(model=_StepMTPText())
    original = rt.model.mtp.layers[0].shared_head_head
    rt.model.mtp.layers[0].shared_head_head = LoRALinear(original, rank=2)

    report = _install_draft_lm_head(rt, bits=3, group_size=32, mode="affine")
    wrapped = rt.model.mtp.layers[0].shared_head_head
    logits = wrapped(mx.ones((1, 2, 32), dtype=mx.bfloat16))
    mx.eval(logits)

    assert report["source"] == "step_mtp_shared_head"
    assert report["layers"][0]["wrapper"] == "LoRALinear"
    assert report["layers"][0]["draft_only"]["bits"] == 3
    assert isinstance(wrapped, LoRALinear)
    assert wrapped.base.bits == 3
    assert logits.shape == (1, 2, 64)


def test_install_draft_lm_head_shares_a_rotated_packed_target_head() -> None:
    """Bonsai's 2-bit rotated head is the draft head: never requantized.

    A 4-bit copy would be larger than the target head, and a copy made from
    the raw packed rows would skip the activation transform. The server calls
    this installer for every qwen3_next pack with a draft head, so it must
    not raise on the rotated layer either.
    """

    from mtplx.models.prism_hadamard_qwen35 import HadamardQuantizedLinear

    head = HadamardQuantizedLinear(1024, 64, block=1024)
    rt = SimpleNamespace(model=SimpleNamespace(lm_head=head))

    report = _install_draft_lm_head(rt, bits=4, group_size=64, mode="affine")

    assert rt.model._mtplx_draft_lm_head is head
    assert report["source"] == "rotated_packed_lm_head"
    assert report["reused_existing_quantization"] is True
    assert report["draft_only"]["bits"] == 2
    assert report["draft_only"]["group_size"] == 128
    logits = head(mx.ones((1, 2, 1024), dtype=mx.float16))
    mx.eval(logits)
    assert logits.shape == (1, 2, 64)

