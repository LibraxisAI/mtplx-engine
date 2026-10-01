"""The fused hyper-connection READ at prefill width (2026-09-23).

``mtplx.kernels.hc_prefill`` replaces the eager norm -> mix chain of a
GatedResidual read on prefill forwards of 32 rows or more.  It is on by default
(bit-identical to the eager chain on Flash-Next's real weights, and the whole
prompt state is bit-identical at 4K and 16K), and ``MTPLX_QWEN4_HC_PREFILL_READ=0``
restores the eager chain without ever reaching the kernel.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from mtplx.models import qwen4_exp


def test_hc_prefill_read_is_on_by_default_and_the_switch_turns_it_off(monkeypatch):
    monkeypatch.delenv("MTPLX_QWEN4_HC_PREFILL_READ", raising=False)
    assert qwen4_exp._hc_prefill_read_enabled()
    for on in ("1", "true", "yes", "on"):
        monkeypatch.setenv("MTPLX_QWEN4_HC_PREFILL_READ", on)
        assert qwen4_exp._hc_prefill_read_enabled()
    for off in ("0", "false", "no", "off", " OFF "):
        monkeypatch.setenv("MTPLX_QWEN4_HC_PREFILL_READ", off)
        assert not qwen4_exp._hc_prefill_read_enabled()


def _owner():
    args = SimpleNamespace(hc_count=4, hidden_size=64, hc_lowrank=8, rms_norm_eps=1e-6)
    return qwen4_exp.GatedResidual(args)


def test_hc_prefill_read_switch_off_keeps_the_eager_chain(monkeypatch):
    import mtplx.kernels.hc_prefill as hc

    def refuse(owner, hyper_input):  # pragma: no cover - must not be reached
        raise AssertionError("the fused read ran with its switch off")

    monkeypatch.setattr(hc, "hc_prefill_read", refuse)
    monkeypatch.setenv("MTPLX_QWEN4_HC_PREFILL_READ", "0")
    mixed, hyper, inject = _owner()(mx.random.normal((1, 40, 256)).astype(mx.float32))
    mx.eval(mixed, inject)
    assert mixed.shape == (1, 40, 64) and inject.shape == (1, 40, 4)


def test_hc_prefill_read_declining_falls_back_to_the_eager_chain(monkeypatch):
    import mtplx.kernels.hc_prefill as hc

    calls = []

    def decline(owner, hyper_input):
        calls.append(tuple(hyper_input.shape))
        return None

    monkeypatch.setattr(hc, "hc_prefill_read", decline)
    monkeypatch.delenv("MTPLX_QWEN4_HC_PREFILL_READ", raising=False)
    owner = _owner()
    x = mx.random.normal((1, 40, 256)).astype(mx.float32)
    mixed, _, inject = owner(x)
    monkeypatch.setenv("MTPLX_QWEN4_HC_PREFILL_READ", "0")
    want_mixed, _, want_inject = owner(x)
    mx.eval(mixed, inject, want_mixed, want_inject)
    assert calls == [(1, 40, 256)]
    assert mx.array_equal(mixed, want_mixed).item()
    assert mx.array_equal(inject, want_inject).item()


def test_hc_prefill_read_is_not_consulted_below_prefill_width(monkeypatch):
    import mtplx.kernels.hc_prefill as hc

    def refuse(owner, hyper_input):  # pragma: no cover - must not be reached
        raise AssertionError("the fused read ran on a decode-width forward")

    monkeypatch.setattr(hc, "hc_prefill_read", refuse)
    monkeypatch.delenv("MTPLX_QWEN4_HC_PREFILL_READ", raising=False)
    rows = qwen4_exp._HC_COMPILE_MIN_ROWS - 1
    mixed, _, inject = _owner()(mx.random.normal((1, rows, 256)).astype(mx.float32))
    mx.eval(mixed, inject)
    assert mixed.shape == (1, rows, 64)
