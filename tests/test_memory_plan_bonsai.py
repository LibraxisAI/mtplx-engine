"""Planner verdicts for the 8.2 GiB Bonsai pack, before and after the measured rule.

Before the 2026-09-21 measurement the 16 GiB class REFUSED this pack
(``model_fits`` False, the 4096 field a fallback, not an admission): the
12 GiB engine budget could not fund 8.2 GiB of weights + the 3 GiB runtime
transient + the 1 GiB bank floor + one KV block. The measurement
(``scripts/bonsai_memory_table.py``, outputs/release-2114/speed/
bonsai-memory-1810/memory.json) showed the peak WITHOUT a resident bank is
weights + dense KV + 3.056 to 3.075 GiB in every completed row, so on the
16 GiB class 4K peaks at 11.55 GiB and 8K at 11.78 to 11.80 GiB under the
12 GiB budget while 16K peaks at 12.11 GiB over it; the q8 KV setting did
not lower any of those peaks (the 16K peaks were byte-identical). The
tight-machine rule therefore admits 8192 tokens on 16 GiB with the bank
floor at zero and the KV counted at its dense width. Every other class is
unchanged. The rule is offered to packs no heavier than this one and to a
pack that stamps its own measurement (``memory_evidence`` in its runtime
contract).

Actual pack bytes (including the MTP sidecar) are read by the measurement
script; these deliberately pin the requested round-number 8.2 GiB baseline.
"""

import pytest

from mtplx.memory_plan import (
    BANK_FLOOR_BYTES,
    GIB,
    RUNTIME_TRANSIENTS_BYTES,
    TIGHT_MACHINE_MARGIN_BYTES,
    TIGHT_MACHINE_MAX_WEIGHTS_BYTES,
    bank_dynamic_ceiling,
    describe_plan,
    measured_on_tight_machine,
    plan_memory,
)

BONSAI_WEIGHTS = 8_804_682_956  # floor(8.2 * GiB), including all weights
BONSAI_KV_PER_TOKEN = 65_536  # 16 full-attention layers, K+V, 4 heads, dim 256, fp16


def _plan(ram: int, quant: str = "off", *, tight_machine_measured: bool = True, **kw):
    # The published pack stamps its measured memory table.
    return plan_memory(
        total_ram_bytes=ram * GIB,
        model_weights_bytes=BONSAI_WEIGHTS,
        kv_bytes_per_token=BONSAI_KV_PER_TOKEN,
        kv_quantization=quant,
        model_max_context=262_144,
        tight_machine_measured=tight_machine_measured,
        **kw,
    )


def _evidence(weights: int = BONSAI_WEIGHTS, **row) -> dict:
    """The shape scripts/bonsai_memory_table.py stamps into the contract."""

    measured = {
        "completed": True,
        "within_engine_budget": True,
        "planner": {"tight_machine": True},
        **row,
    }
    return {"memory_evidence": {"pack": {"weights_bytes": weights}, "rows": [measured]}}


@pytest.mark.parametrize("ram,budget,admitted,off_context,q8_context", [
    (16, 12, True, 8192, 8192),  # was (16, 12, False, 4096, 4096) before the rule
    (18, 13.5, True, 20480, 36864),
    (24, 18, True, 94208, 172032),
    (32, 24, True, 192512, 262144),
])
@pytest.mark.parametrize("quant", ["off", "q8"])
def test_bonsai_planner_verdict(ram, budget, admitted, off_context, q8_context, quant):
    plan = _plan(ram, quant)
    assert plan.available
    assert plan.usable_bytes == int(budget * GIB)
    assert plan.model_fits is admitted
    assert plan.context_window_resolved == (off_context if quant == "off" else q8_context)
    if ram == 16:
        # Tight machine: the KV rate is dense whatever the paged setting says.
        assert plan.kv_bytes_per_token_effective == 65_536
    else:
        assert plan.kv_bytes_per_token_effective == (65_536 if quant == "off" else 36_044)
        assert plan.tight_machine is False
        assert plan.bank_floor_bytes == BANK_FLOOR_BYTES
        assert plan.runtime_transients_bytes == RUNTIME_TRANSIENTS_BYTES


@pytest.mark.parametrize("quant", ["off", "q8"])
def test_16_gib_admits_bonsai_with_the_bank_floor_at_zero(quant):
    plan = _plan(16, quant, dense_decode_ceiling=32_768)
    usable = 12 * GIB
    transients = RUNTIME_TRANSIENTS_BYTES + TIGHT_MACHINE_MARGIN_BYTES
    assert plan.model_fits and plan.tight_machine
    assert plan.bank_floor_bytes == 0
    assert plan.runtime_transients_bytes == transients
    # 12 GiB - 8.2 GiB - 3.25 GiB = 0.55 GiB of KV at 64 KiB per token:
    # 8.5K tokens, block-aligned down to 8192 (the measured 8K row peaked
    # 0.2 GiB under the budget; 12K would not).
    assert plan.context_window_fit == 8192
    assert plan.bank_idle_max_bytes == usable - BONSAI_WEIGHTS - transients
    assert plan.bank_steady_bytes == (
        usable - BONSAI_WEIGHTS - transients - 8192 * BONSAI_KV_PER_TOKEN
    )
    assert plan.headroom_bytes == 0
    # Steady state fits the envelope with the margin, not the bank floor.
    assert (
        BONSAI_WEIGHTS + plan.kv_reserve_bytes + plan.bank_steady_bytes + transients
        <= usable
    )
    assert any(note.startswith("tight machine") for note in plan.notes)
    assert not any("model does not fit" in note for note in plan.notes)
    assert "tight machine" in describe_plan(plan)
    assert "MODEL DOES NOT FIT" not in describe_plan(plan)
    data = plan.to_dict()
    assert data["tight_machine"] is True
    assert data["bank_floor_bytes"] == 0
    assert data["runtime_transients_bytes"] == transients


def test_tight_machine_dynamic_ceiling_can_reach_zero():
    plan = _plan(16, dense_decode_ceiling=32_768)
    assert bank_dynamic_ceiling(plan, 0) == plan.bank_idle_max_bytes
    # An 8K live KV (0.5 GiB) leaves 22 MiB; a 1 GiB working set leaves nothing.
    assert bank_dynamic_ceiling(plan, 8192 * BONSAI_KV_PER_TOKEN) == plan.bank_steady_bytes
    assert bank_dynamic_ceiling(plan, 1 * GIB) == 0
    # An observed spike below the plan's own transient does not loosen it.
    assert bank_dynamic_ceiling(plan, 0, transient_bytes=RUNTIME_TRANSIENTS_BYTES) == (
        plan.bank_idle_max_bytes
    )


def test_tight_machine_still_refuses_what_the_margin_cannot_fund():
    # 9.5 GiB of weights: 12 - 9.5 - 3.25 < 0 even without the bank floor.
    plan = plan_memory(
        total_ram_bytes=16 * GIB,
        model_weights_bytes=int(9.5 * GIB),
        kv_bytes_per_token=BONSAI_KV_PER_TOKEN,
        model_max_context=262_144,
        tight_machine_measured=True,
    )
    assert plan.model_fits is False
    assert plan.tight_machine is False
    assert plan.bank_floor_bytes == BANK_FLOOR_BYTES
    assert plan.context_window_resolved == 4096  # fallback, not an admission
    assert any("model does not fit" in note for note in plan.notes)


def test_tight_machine_counts_q8_kv_at_the_dense_width():
    # Measured: the 16K peaks on the 16 GiB class were byte-identical for
    # KV off and q8, so q8 must not buy phantom context on a tight machine.
    assert _plan(16, "q8").context_window_fit == _plan(16, "off").context_window_fit


def test_a_machine_that_funds_the_bank_floor_is_not_tight():
    plan = _plan(18, dense_decode_ceiling=32_768)
    assert plan.model_fits and not plan.tight_machine
    assert plan.bank_floor_bytes == BANK_FLOOR_BYTES
    assert plan.bank_steady_bytes >= BANK_FLOOR_BYTES


def test_packs_inside_the_measured_envelope_are_admitted_tight():
    # Qwen3.5-9B Optimized Speed: 8.08 GiB of weights (MTP and vision
    # included) and 8 attention layers at 32 KiB per token. 12 - 8.08 - 3.25
    # leaves 0.67 GiB of KV: 21.9K tokens, block-aligned to 20,480.
    nine_b = plan_memory(
        total_ram_bytes=16 * GIB,
        model_weights_bytes=8_674_988_799,
        kv_bytes_per_token=32_768,
        model_max_context=262_144,
    )
    assert nine_b.model_fits and nine_b.tight_machine
    assert nine_b.context_window_resolved == 20_480
    # The published Bonsai 2 pack sits exactly at the envelope.
    bonsai = plan_memory(
        total_ram_bytes=16 * GIB,
        model_weights_bytes=TIGHT_MACHINE_MAX_WEIGHTS_BYTES,
        kv_bytes_per_token=BONSAI_KV_PER_TOKEN,
        model_max_context=262_144,
    )
    assert bonsai.model_fits and bonsai.tight_machine
    assert bonsai.context_window_resolved == 8192


def test_a_lighter_pack_never_plans_a_smaller_window_than_a_heavier_one():
    # 2026-09-22 on 16 GiB: the MiMo 9B mixed-precision test build (7.53
    # GiB, the Qwen3.5-9B geometry) planned 12,288 tokens and the heavier
    # 8.08 GiB Qwen3.5-9B 6-bit pack 20,480. The lighter one funds the bank
    # floor (12 - 7.53 - 3 - 1 leaves 0.47 GiB of KV), so it never reached
    # the tight rule, which only the heavier pack needed (0.67 GiB of KV).
    def plan(weights: int):
        return plan_memory(
            total_ram_bytes=16 * GIB,
            model_weights_bytes=weights,
            kv_bytes_per_token=32_768,
            model_max_context=262_144,
        )

    lighter = plan(8_081_087_720)
    heavier = plan(8_674_988_799)
    assert heavier.tight_machine and heavier.context_window_resolved == 20_480
    assert lighter.context_window_resolved >= heavier.context_window_resolved
    # The lighter pack yields its floor too and takes the most the tight
    # rule grants any pack: one block plus the floor net of the margin
    # (0.875 GiB of KV, 28,671 tokens), block-aligned to 24,576.
    assert lighter.tight_machine and lighter.bank_floor_bytes == 0
    assert lighter.context_window_resolved == 24_576
    assert any("the window stops at 24576 tokens" in note for note in lighter.notes)
    # A pack whose floor-funded window already reaches that keeps its floor
    # and its window (the 7.10 GiB mixed 4-bit build).
    m0 = plan(7_626_136_774)
    assert not m0.tight_machine and m0.bank_floor_bytes == BANK_FLOOR_BYTES
    assert m0.context_window_resolved == 28_672


def test_a_pack_heavier_than_bonsai_is_admitted_tight_only_with_its_own_table():
    # A 16 GB pack on the one budget where only the tight arithmetic fits it
    # (18.75 GiB: 0.6 GiB of KV after weights and the tight transient).
    heavy = dict(
        total_ram_bytes=25 * GIB,
        model_weights_bytes=16_000_000_000,
        kv_bytes_per_token=BONSAI_KV_PER_TOKEN,
        model_max_context=262_144,
    )
    plan = plan_memory(**heavy)
    assert plan.model_fits is False
    assert plan.tight_machine is False
    assert plan.context_window_resolved == 4096  # fallback, not an admission
    measured = plan_memory(**heavy, tight_machine_measured=True)
    assert measured.model_fits and measured.tight_machine
    assert measured.context_window_resolved == 8192


def test_measured_on_tight_machine_reads_the_stamped_memory_table():
    assert measured_on_tight_machine(_evidence(), BONSAI_WEIGHTS) is True
    # Another pack's table, a row over its budget, a row the rule did not
    # admit, and no table at all never count.
    assert measured_on_tight_machine(_evidence(weights=1), BONSAI_WEIGHTS) is False
    assert measured_on_tight_machine(_evidence(within_engine_budget=False), BONSAI_WEIGHTS) is False
    assert measured_on_tight_machine(_evidence(planner={"tight_machine": False}), BONSAI_WEIGHTS) is False
    assert measured_on_tight_machine({}, BONSAI_WEIGHTS) is False
    assert measured_on_tight_machine(None, BONSAI_WEIGHTS) is False
