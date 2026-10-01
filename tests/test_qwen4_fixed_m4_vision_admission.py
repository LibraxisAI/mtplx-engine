"""Image requests at the compiled fixed-M4 verify lane's door.

Until 2.11.4 one image anywhere in the retained history sent the whole request
to the eager verifier, for every later text token. The admission now hands the
verify bank the request's rotary delta, and keeps eager only what the delta
cannot express, each with a named reason in the request record. No model here:
the admission is host arithmetic over the prompt ids and the splice.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mtplx.generation as gen
from mtplx import demotions

PAD = 900
RATIO = 4


class _Table:
    """The one thing the admission reads off a position table: its length."""

    def __init__(self, tokens: int) -> None:
        self.shape = (3, tokens)


def _rt(*, lane: bool = True, ratio: int = RATIO):
    args = SimpleNamespace(
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 12,
        num_key_value_heads=2,
        head_dim=256,
        indexer_head_dim=128,
        indexer_compress_ratio=ratio,
    )
    return SimpleNamespace(
        qwen4_fixed_m4_compiled_verify=lane,
        model=SimpleNamespace(args=args),
        metal_memory_limit_bytes=None,
    )


def _prompt(tokens: int, *, image_end: int, image_tokens: int = 16) -> list[int]:
    """``tokens`` ids whose last image placeholder sits at ``image_end``."""

    start = image_end - image_tokens + 1
    ids = list(range(1, tokens + 1))
    ids[start : image_end + 1] = [PAD] * image_tokens
    return ids


def _splice(ids, *, delta: int | None = -12, table: bool = True, dense=None, images=1):
    return SimpleNamespace(
        image_pad_token_id=PAD,
        mrope_table=_Table(len(ids)) if table else None,
        mrope_delta=0 if delta is None else delta,
        dense_mrope=dense,
        pad_counts=tuple([16] * images),
    )


def _admit(rt, ids, splice, **overrides):
    receipt: dict = {}
    kwargs = dict(
        vision_splice=splice,
        prompt_ids=ids,
        bank_key_ids=ids,
        verify_strategy="batched",
        compiled_mode="on",
        max_tokens=256,
        cached_tokens=0,
        speculative_depth=3,
        session_bank=None,
        receipt=receipt,
    )
    kwargs.update(overrides)
    admitted, rope_delta = gen._qwen4_fixed_m4_admission(rt, **kwargs)
    return admitted, rope_delta, receipt


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in (
        "MTPLX_QWEN4_VISION_COMPILED_VERIFY",
        "MTPLX_QWEN4_VISION_QSA",
        "MTPLX_STATE_REBASE_EVERY",
        "MTPLX_QWEN4_FIXED_M4_MAX_CONTEXT",
    ):
        monkeypatch.delenv(name, raising=False)
    demotions.reset()
    yield
    demotions.reset()


def test_a_text_request_takes_the_text_trace_and_says_so():
    admitted, rope_delta, receipt = _admit(_rt(), list(range(1, 65)), None)
    assert admitted is True and rope_delta is None
    expected = {
        "positions": "text",
        "rope_delta": None,
        "images": 0,
        "requested_depth": 3,
        "engaged": True,
        "reason": "admitted",
    }
    # (The memory gate adds its own byte counts, which depend on the machine.)
    assert {key: receipt[key] for key in expected} == expected
    assert demotions.snapshot()["total"] == 0


def test_an_image_request_is_admitted_with_its_delta():
    ids = _prompt(64, image_end=40)
    admitted, rope_delta, receipt = _admit(_rt(), ids, _splice(ids, delta=-990))
    assert admitted is True and rope_delta == -990
    assert receipt["positions"] == "vision_delta"
    assert receipt["rope_delta"] == -990
    assert receipt["images"] == 1
    assert receipt["engaged"] is True and receipt["reason"] == "admitted"
    assert demotions.snapshot()["total"] == 0  # nothing left the lane


def test_an_image_request_at_sequence_positions_takes_the_text_trace():
    """No table and delta 0 (the table could not be built): the request was
    prefilled at plain positions and decodes at them."""

    ids = _prompt(64, image_end=63)  # even ending ON the image: no table, no rule
    admitted, rope_delta, receipt = _admit(
        _rt(), ids, _splice(ids, delta=None, table=False)
    )
    assert admitted is True and rope_delta is None
    assert receipt["positions"] == "vision_sequential"
    assert receipt["rope_delta"] is None and receipt["images"] == 1


def test_sequential_image_positions_do_not_bypass_the_rebase_refusal(monkeypatch):
    ids = _prompt(64, image_end=63)
    splice = _splice(ids, delta=0, table=False)
    monkeypatch.setenv("MTPLX_STATE_REBASE_EVERY", "64")
    admitted, delta, receipt = _admit(_rt(), ids, splice)
    assert (admitted, delta) == (False, None)
    assert receipt["positions"] == "vision_sequential"
    assert receipt["reason"] == "vision_state_rebase"
    monkeypatch.delenv("MTPLX_STATE_REBASE_EVERY")
    admitted, delta, receipt = _admit(_rt(), ids, splice)
    assert (admitted, delta) == (True, None)
    assert receipt["reason"] == "admitted"


@pytest.mark.parametrize(
    ("tokens", "image_end", "refused"),
    [
        (64, 59, False),  # block starts at 64: nothing of the prompt is in it
        (65, 63, False),  # block starts at 64, the last image row is 63
        (65, 64, True),  # the block starts ON the last image row
        (67, 64, True),  # ...and stays refused while the block start is 64
        (67, 63, False),
        (68, 66, False),  # a whole text block closed the prompt: start is 68
    ],
)
def test_the_tail_block_rule(tokens, image_end, refused):
    ids = _prompt(tokens, image_end=image_end)
    assert max(i for i, t in enumerate(ids) if t == PAD) == image_end
    admitted, rope_delta, receipt = _admit(_rt(), ids, _splice(ids))
    assert admitted is (not refused)
    if refused:
        assert rope_delta is None
        assert receipt["reason"] == "vision_tail_block_in_image"
        assert receipt["engaged"] is False
        assert receipt["positions"] == "vision_delta" and receipt["rope_delta"] == -12
        snap = demotions.snapshot()
        assert snap["counts"]["vision_request_eager_verify"] == 1
        assert "indexer block" in snap["reasons"]["vision_request_eager_verify"]
    else:
        assert rope_delta == -12 and receipt["reason"] == "admitted"


def test_the_rule_follows_the_models_own_block_width():
    ids = _prompt(66, image_end=64)
    assert _admit(_rt(ratio=4), ids, _splice(ids))[0] is False  # block start 64
    assert _admit(_rt(ratio=2), ids, _splice(ids))[0] is True  # block start 66
    # A geometry that cannot be read is never guessed at.
    assert _admit(_rt(ratio=0), ids, _splice(ids))[2]["reason"] == (
        "vision_tail_block_in_image"
    )


@pytest.mark.parametrize(
    ("env", "value", "reason"),
    [
        ("MTPLX_QWEN4_VISION_COMPILED_VERIFY", "0", "vision_kill_switch"),
        ("MTPLX_QWEN4_VISION_QSA", "0", "vision_qsa_disabled"),
        ("MTPLX_STATE_REBASE_EVERY", "64", "vision_state_rebase"),
    ],
)
def test_settings_that_keep_an_image_request_eager(monkeypatch, env, value, reason):
    monkeypatch.setenv(env, value)
    ids = _prompt(64, image_end=40)
    admitted, rope_delta, receipt = _admit(_rt(), ids, _splice(ids))
    assert (admitted, rope_delta) == (False, None)
    assert receipt["reason"] == reason and receipt["engaged"] is False
    assert demotions.snapshot()["counts"]["vision_request_eager_verify"] == 1
    # Text requests never notice any of them.
    demotions.reset()
    assert _admit(_rt(), list(range(1, 65)), None)[0] is True
    assert demotions.snapshot()["total"] == 0


def test_the_kill_switch_is_the_previous_behaviour_for_every_image_request(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_VISION_COMPILED_VERIFY", "0")
    ids = _prompt(64, image_end=40)
    sequential = _splice(ids, delta=None, table=False)
    assert _admit(_rt(), ids, sequential)[2]["reason"] == "vision_kill_switch"


def test_the_default_is_on(monkeypatch):
    ids = _prompt(64, image_end=40)
    for value in ("1", "true"):
        monkeypatch.setenv("MTPLX_QWEN4_VISION_COMPILED_VERIFY", value)
        assert _admit(_rt(), ids, _splice(ids))[0] is True
    monkeypatch.delenv("MTPLX_QWEN4_VISION_COMPILED_VERIFY")
    assert _admit(_rt(), ids, _splice(ids))[0] is True


def test_a_dense_path_splice_and_a_short_table_stay_eager(monkeypatch):
    ids = _prompt(64, image_end=40)
    monkeypatch.setattr(gen, "_dense_mrope_state_of", lambda splice: object())
    assert _admit(_rt(), ids, _splice(ids))[2]["reason"] == "vision_dense_mrope"
    monkeypatch.undo()
    short = _splice(ids)
    short.mrope_table = _Table(len(ids) - 1)
    assert _admit(_rt(), ids, short)[2]["reason"] == "vision_table_length_mismatch"


def test_positions_are_checked_before_memory_is_touched(monkeypatch):
    """The memory gate may evict idle bank entries; a request that stays eager
    anyway must not cost that."""

    def unexpected(*_args, **_kwargs):
        raise AssertionError("the memory gate ran for a request that stays eager")

    monkeypatch.setattr(gen, "_qwen4_fixed_m4_lane_fits", unexpected)
    ids = _prompt(65, image_end=64)
    assert _admit(_rt(), ids, _splice(ids))[2]["reason"] == "vision_tail_block_in_image"


def test_the_memory_gate_protects_the_sessions_own_image_entry(monkeypatch):
    """Image entries are keyed by surrogate ids (one per image row, derived
    from the pixels); the model-input ids would protect nothing."""

    ids = _prompt(64, image_end=40)
    keyed = [token if token != PAD else (1 << 62) + index for index, token in enumerate(ids)]
    gib = 1024**3
    state = {"live": 99 * gib}
    monkeypatch.setattr(gen, "_mlx_live_memory_bytes", lambda: state["live"])
    monkeypatch.setattr(gen, "_mlx_release_allocator_cache", lambda: 0)
    seen = []

    class Bank:
        total_nbytes = 8 * gib

        def shrink_for_admission(self, target, *, protect_tokens, reason):
            seen.append(list(protect_tokens))
            state["live"] = 60 * gib
            return 1, 0

    rt = _rt()
    rt.metal_memory_limit_bytes = 96 * gib
    admitted, rope_delta, receipt = _admit(
        rt, ids, _splice(ids), bank_key_ids=keyed, session_bank=Bank()
    )
    assert admitted is True and rope_delta == -12
    assert seen == [keyed] and seen[0] != ids
    assert receipt["chain_entries_evicted"] == 1


def test_a_family_without_the_lane_records_nothing():
    ids = _prompt(64, image_end=64)
    admitted, rope_delta, receipt = _admit(_rt(lane=False), ids, _splice(ids))
    assert (admitted, rope_delta, receipt) == (False, None, {})
    assert demotions.snapshot()["total"] == 0


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"verify_strategy": "capture_commit"}, "verify_strategy_not_batched"),
        ({"compiled_mode": "off"}, "compiled_verify_off"),
        ({"max_tokens": 0}, "no_decode_budget"),
        ({"compiled_mode": "parity"}, "compiled_verify_parity_mode"),
        ({"speculative_depth": 2}, "depth_below_compiled_window"),
    ],
)
def test_the_record_is_filled_on_every_path(overrides, reason):
    ids = _prompt(64, image_end=40)
    admitted, rope_delta, receipt = _admit(_rt(), ids, _splice(ids), **overrides)
    assert (admitted, rope_delta) == (False, None)
    assert receipt["reason"] == reason and receipt["engaged"] is False
    assert receipt["positions"] == "vision_delta" and receipt["images"] == 1


def test_parity2_is_admitted_because_the_installed_lane_has_that_instrument():
    ids = _prompt(64, image_end=40)
    admitted, rope_delta, _ = _admit(_rt(), ids, _splice(ids), compiled_mode="parity2")
    assert admitted is True and rope_delta == -12


def test_images_are_counted_from_the_splice_or_from_the_prompt():
    ids = _prompt(96, image_end=40)
    ids[60:76] = [PAD] * 16
    assert _admit(_rt(), ids, _splice(ids, images=2))[2]["images"] == 2
    bare = _splice(ids)
    bare.pad_counts = None
    assert _admit(_rt(), ids, bare)[2]["images"] == 2  # two runs of placeholders


def test_only_a_context_roped_request_is_kept_off_the_other_compiled_routes():
    ids = _prompt(64, image_end=40)
    assert gen._vision_rope_request(None) is False
    assert gen._vision_rope_request(_splice(ids, delta=None, table=False)) is False
    assert gen._vision_rope_request(_splice(ids)) is True
    assert gen._vision_rope_request(_splice(ids, delta=-3, table=False)) is True


def test_the_generation_loop_is_wired_to_the_admission():
    import inspect

    source = inspect.getsource(gen.generate_mtpk)
    admission = source.index("_qwen4_fixed_m4_admission(")
    install = source.index("compiled_verify_bank.install_fixed_m4(")
    assert 0 < admission < install
    assert "bank_key_ids=bank_commit_ids" in source[admission:install]
    assert "rope_delta=fixed_m4_rope_delta" in source[install : install + 900]
    # The old blanket guard is gone, and so is its one-line reason.
    assert "vision_splice is None\n        and _qwen4_fixed_m4" not in source
    assert "fixed_m4_admission=fixed_m4_admission" in source


def test_the_admission_decides_roped_requests_on_the_bank_keys_predicate():
    """One predicate says whether a request's rows are positioned by its
    table and delta: the attention scope opens on it, the session bank key is
    salted on it, and the compiled route admits on it. Three readers of one
    answer cannot drift apart."""

    from mtplx.vision.splice import mrope_rope_state

    ids = list(range(1, 25)) + [PAD] * 16 + list(range(30, 40))
    for splice in (
        None,
        _splice(ids),
        _splice(ids, delta=None, table=False),
        _splice(ids, delta=-3, table=False),
        _splice(ids, delta=None, table=True),
    ):
        roped = mrope_rope_state(splice)
        assert gen._vision_rope_request(splice) is (roped is not None)
        if splice is None:
            continue
        verdict = gen._qwen4_vision_compiled_verify_admission(_rt(), splice, ids)
        assert (verdict["positions"] == "vision_delta") is (roped is not None)
        assert verdict["rope_delta"] == (roped[1] if roped is not None else None)
