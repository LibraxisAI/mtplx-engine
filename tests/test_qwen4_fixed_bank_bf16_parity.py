"""The fixed QSA bank against the stock cache in bfloat16, bit for bit.

bfloat16 weights and hidden states (the served dtype), CPU as the parity surface like the other QSA cache tests.
The stock ``QSACache`` lane is the oracle throughout: promotion, verify-width steps that complete a pooled block,
steps that do not, a rejected window that is rolled back, growth, and the way back out of the fixed bank.
"""

import mlx.core as mx
import mlx.utils
import pytest

import mtplx.graphbank as graphbank
from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs

PREFILL = 13  # odd on purpose: the first verify step straddles a block edge (ratio 2)
STEP = 4


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )


@pytest.fixture()
def attn():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(3)
    layer = Attention(_tiny_args())
    layer.update(mlx.utils.tree_map(lambda p: p.astype(mx.bfloat16), layer.parameters()))
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(prev)


def _hidden(tokens: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, 64)).astype(mx.bfloat16)


def _promote(attn, cache):
    promoted, failures = graphbank.promote_kv_cache_offsets(
        cache, reserve_tokens=STEP, initial_reserve_tokens=32
    )
    assert promoted == 1 and failures == {}
    assert isinstance(cache[0], graphbank.TensorOffsetQSACache)
    return cache[0]


def _same(a: mx.array, b: mx.array) -> bool:
    return a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


def test_every_step_matches_the_stock_lane_bit_for_bit(attn):
    """Steps that complete a block, steps that do not, and single rows."""

    steps = [_hidden(n, 10 + i) for i, n in enumerate((STEP, STEP, 1, 3, STEP, 1, 1, STEP))]
    fixed = [QSACache(compress_ratio=attn.indexer.ratio)]
    stock = QSACache(compress_ratio=attn.indexer.ratio)
    x_pre = _hidden(PREFILL, 2)
    attn(x_pre, fixed[0])
    attn(x_pre, stock)
    _promote(attn, fixed)
    for x in steps:
        out_fixed = attn(x, fixed[0])
        out_stock = attn(x, stock)
        assert _same(out_fixed, out_stock)
    valid = fixed[0].size() // attn.indexer.ratio
    assert _same(
        fixed[0].pooled[:, :valid].astype(mx.float32),
        stock.pooled[:, :valid].astype(mx.float32),
    )


def test_a_rejected_window_leaves_no_trace(attn):
    """A block the rejected tokens completed is rewritten by the accepted ones."""

    x_pre = _hidden(PREFILL, 4)
    fixed = [QSACache(compress_ratio=attn.indexer.ratio)]
    attn(x_pre, fixed[0])
    _promote(attn, fixed)
    attn(_hidden(STEP, 5), fixed[0])  # drafted window
    assert fixed[0].trim(3) == 3  # one token accepted, three rejected
    x_next = _hidden(STEP, 6)
    out = attn(x_next, fixed[0])

    stock = QSACache(compress_ratio=attn.indexer.ratio)
    attn(x_pre, stock)
    attn(_hidden(STEP, 5)[:, :1], stock)
    golden = attn(x_next, stock)
    assert _same(out, golden)


def test_the_way_back_out_of_the_bank_is_exact(attn):
    x_pre = _hidden(PREFILL, 7)
    fixed = [QSACache(compress_ratio=attn.indexer.ratio)]
    stock = QSACache(compress_ratio=attn.indexer.ratio)
    attn(x_pre, fixed[0])
    attn(x_pre, stock)
    bank = _promote(attn, fixed)
    for seed in (8, 9):
        x = _hidden(STEP, seed)
        attn(x, bank)
        attn(x, stock)
    entry = bank.demote()
    assert isinstance(entry, QSACache)
    assert entry.pooled.dtype == stock.pooled.dtype == mx.bfloat16
    valid = entry.pooled_len
    assert valid == stock.pooled_len
    assert _same(entry.pooled[:, :valid], stock.pooled[:, :valid])
    x_after = _hidden(STEP, 11)
    assert _same(attn(x_after, entry), attn(x_after, stock))


# ---------------------------------------------------------------------------
# Image requests: the bank owns its rotary origin.
#
# The oracle is the eager verifier as it runs an image request: the stock
# ``QSACache`` inside the request's position scope (``vision_rope(table,
# delta)``). The candidate is the promoted bank carrying ``rope_delta``, run
# with NO scope open, because the fixed lane never reads the request context.
# Bit for bit, with the rope glue kernels and the op diet off and on.
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from mtplx import qwen4_verify_glue, runtime_options  # noqa: E402
from mtplx.attention_context import vision_rope  # noqa: E402
from mtplx.vision.mrope import build_mrope_positions  # noqa: E402

PAD = 99
RATIO = 4  # the served indexer ratio
IMAGE_TOKENS = 16  # a (1, 8, 8) patch grid at merge 2: 4 x 4 rows, 4 positions
STEPS = (STEP, STEP, 1, 3, STEP, 1, 1, STEP)

_LANES = {
    "stock": {},
    "opdiet": {"MTPLX_QWEN4_OPDIET": "1"},
    "glue": {"MTPLX_QWEN4_VERIFY_GLUE": "1"},
    "glue_opdiet": {"MTPLX_QWEN4_VERIFY_GLUE": "1", "MTPLX_QWEN4_OPDIET": "1"},
}


def _vision_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=8,
        indexer_compress_ratio=RATIO,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    )


@pytest.fixture(params=list(_LANES))
def vision_attn(request):
    """One bf16 QSA layer with an M-RoPE contract, per rope lane.

    The glue lanes run the two fused rope kernels, which are Metal kernels,
    so they run on the GPU; the stock and op-diet lanes keep the CPU parity
    surface of the rest of this file.
    """

    env = _LANES[request.param]
    glue = "MTPLX_QWEN4_VERIFY_GLUE" in env
    if glue and not mx.metal.is_available():
        pytest.skip("the rope glue kernels are Metal kernels")
    prev = mx.default_device()
    mx.set_default_device(mx.gpu if glue else mx.cpu)
    runtime_options.reset_qwen4_verify_glue_cache(env=env)
    runtime_options.reset_qwen4_opdiet_cache(env=env)
    qwen4_verify_glue.reset_for_tests()
    try:
        mx.random.seed(3)
        layer = Attention(_vision_args())
        layer.update(
            mlx.utils.tree_map(lambda p: p.astype(mx.bfloat16), layer.parameters())
        )
        mx.eval(layer.parameters())
        assert layer._mrope_axes is not None and layer.indexer._mrope_axes is not None
        if glue:
            report = qwen4_verify_glue.install([(0, layer)], rows=STEP)
            # A glue lane whose probe disabled an item would silently test the
            # stock chain twice.
            assert report["items"]["qsa_rope"]["installed"], report
            assert report["items"]["qsa_rope_idx"]["installed"], report
        yield layer
    finally:
        runtime_options.reset_qwen4_verify_glue_cache(env={})
        runtime_options.reset_qwen4_opdiet_cache(env={})
        qwen4_verify_glue.reset_for_tests()
        mx.set_default_device(prev)


def _image_prompt(tokens: int, tail: int):
    """[text | one image | ``tail`` text tokens]: ids, table [3, n], delta."""

    ids = [1] * (tokens - IMAGE_TOKENS - tail) + [PAD] * IMAGE_TOKENS + [2] * tail
    table, delta = build_mrope_positions(
        ids, image_token_id=PAD, image_grids=[(1, 8, 8)], spatial_merge_size=2
    )
    return ids, mx.array(table), int(delta)


def _prefilled(attn, tokens: int, tail: int, seed: int):
    """The same image prompt prefilled into a stock cache and a promoted bank."""

    ids, table, delta = _image_prompt(tokens, tail)
    x_pre = _hidden(tokens, seed)
    stock = QSACache(compress_ratio=RATIO)
    fixed = [QSACache(compress_ratio=RATIO)]
    with vision_rope(table, delta):
        attn(x_pre, stock)
        attn(x_pre, fixed[0])
    bank = _promote(attn, fixed)
    return ids, table, delta, stock, bank


def _admission(ids, table, delta):
    import mtplx.generation as generation

    rt = SimpleNamespace(
        model=SimpleNamespace(args=SimpleNamespace(indexer_compress_ratio=RATIO))
    )
    splice = SimpleNamespace(
        image_pad_token_id=PAD,
        mrope_table=table,
        mrope_delta=delta,
        dense_mrope=None,
        pad_counts=(IMAGE_TOKENS,),
    )
    return generation._qwen4_vision_compiled_verify_admission(rt, splice, ids)


@pytest.mark.parametrize("tokens", [40, 41, 42, 43])  # n % 4 of 0, 1, 2 and 3
def test_image_request_steps_match_the_eager_verifier_in_its_scope(vision_attn, tokens):
    """Block-completing steps, steps that complete nothing, single rows."""

    attn = vision_attn
    ids, table, delta, stock, bank = _prefilled(attn, tokens, tail=11, seed=2)
    assert _admission(ids, table, delta)["refusal"] is None
    assert delta == 4 - IMAGE_TOKENS  # the image takes 4 positions for 16 rows
    bank.rope_delta = delta
    for i, n in enumerate(STEPS):
        x = _hidden(n, 10 + i)
        with vision_rope(table, delta):
            golden = attn(x, stock)
        assert _same(attn(x, bank), golden), (tokens, i)
    end = bank.size()
    assert end == stock.offset == tokens + sum(STEPS)
    # The state the next round reads: rotated keys, values, raw and pooled keys.
    assert _same(bank.kv.keys[:, :, :end], stock.kv.keys[:, :, :end])
    assert _same(bank.kv.values[:, :, :end], stock.kv.values[:, :, :end])
    assert _same(bank.raw_keys[:, :end], stock.raw_keys[:, :end])
    valid = end // RATIO
    assert stock.pooled_len == valid
    assert _same(bank.pooled[:, :valid], stock.pooled[:, :valid])


def test_image_request_copy_block_widths_complete_several_blocks_at_once(vision_attn):
    """Copy rounds run eager over the promoted bank at widths 9 to 25: one
    forward completes up to seven pooled blocks, each rotated at its own
    start plus the delta."""

    attn = vision_attn
    _ids, table, delta, stock, bank = _prefilled(attn, 42, tail=10, seed=21)
    bank.rope_delta = delta
    for i, n in enumerate((9, STEP, 25, 3, 17)):
        bank.ensure_capacity(bank.size() + n)  # what reserve_fixed_m4_window does
        x = _hidden(n, 30 + i)
        with vision_rope(table, delta):
            golden = attn(x, stock)
        assert _same(attn(x, bank), golden), (i, n)
    valid = bank.size() // RATIO
    assert valid == stock.pooled_len
    assert _same(bank.pooled[:, :valid], stock.pooled[:, :valid])


@pytest.mark.parametrize("tokens", [44, 47])  # n % 4 of 0 and 3
def test_two_images_the_bank_carries_the_delta_of_the_whole_prompt(vision_attn, tokens):
    """Between two images the positions run at the first image's offset; past
    the last one they run at the sum. Decode only ever sees the sum, so one
    delta on the bank is still the eager route bit for bit."""

    import mtplx.generation as generation

    attn = vision_attn
    small = 4  # a (1, 4, 4) grid: 2 x 2 rows, 2 positions
    tail = 9
    head = tokens - IMAGE_TOKENS - 3 - small - tail
    ids = [1] * head + [PAD] * IMAGE_TOKENS + [3] * 3 + [PAD] * small + [2] * tail
    built = build_mrope_positions(
        ids,
        image_token_id=PAD,
        image_grids=[(1, 8, 8), (1, 4, 4)],
        spatial_merge_size=2,
    )
    table, delta = mx.array(built[0]), int(built[1])
    assert delta == (4 - IMAGE_TOKENS) + (2 - small) == -14
    verdict = generation._qwen4_vision_compiled_verify_admission(
        SimpleNamespace(
            model=SimpleNamespace(args=SimpleNamespace(indexer_compress_ratio=RATIO))
        ),
        SimpleNamespace(
            image_pad_token_id=PAD,
            mrope_table=table,
            mrope_delta=delta,
            dense_mrope=None,
            pad_counts=(IMAGE_TOKENS, small),
        ),
        ids,
    )
    assert verdict == {
        "positions": "vision_delta",
        "rope_delta": -14,
        "images": 2,
        "refusal": None,
    }

    x_pre = _hidden(tokens, 40)
    stock = QSACache(compress_ratio=RATIO)
    fixed = [QSACache(compress_ratio=RATIO)]
    with vision_rope(table, delta):
        attn(x_pre, stock)
        attn(x_pre, fixed[0])
    bank = _promote(attn, fixed)
    bank.rope_delta = delta
    for i, n in enumerate(STEPS):
        x = _hidden(n, 50 + i)
        with vision_rope(table, delta):
            golden = attn(x, stock)
        assert _same(attn(x, bank), golden), (tokens, i)
    end = bank.size()
    assert _same(bank.kv.keys[:, :, :end], stock.kv.keys[:, :, :end])
    valid = end // RATIO
    assert stock.pooled_len == valid
    assert _same(bank.pooled[:, :valid], stock.pooled[:, :valid])


def test_image_request_a_rejected_window_leaves_no_trace(vision_attn):
    attn = vision_attn
    _ids, table, delta, stock, bank = _prefilled(attn, 43, tail=11, seed=4)
    bank.rope_delta = delta
    drafted = _hidden(STEP, 5)
    attn(drafted, bank)  # completes the block that started inside the prompt
    assert bank.trim(3) == 3  # one token accepted, three rejected
    x_next = _hidden(STEP, 6)
    out = attn(x_next, bank)

    with vision_rope(table, delta):
        attn(drafted[:, :1], stock)
        golden = attn(x_next, stock)
    assert _same(out, golden)
    valid = bank.size() // RATIO
    assert _same(bank.pooled[:, :valid], stock.pooled[:, :valid])


def test_image_request_the_way_back_out_drops_the_delta_and_is_exact(vision_attn):
    attn = vision_attn
    _ids, table, delta, stock, bank = _prefilled(attn, 40, tail=8, seed=7)
    bank.rope_delta = delta
    for seed in (8, 9):
        x = _hidden(STEP, seed)
        attn(x, bank)
        with vision_rope(table, delta):
            attn(x, stock)
    entry = bank.demote()
    assert isinstance(entry, QSACache)
    # Nothing of the delta travels towards the session bank: the keys are
    # stored already rotated and the next request derives its own delta.
    assert not hasattr(entry, "rope_delta")
    assert entry.offset == stock.offset
    assert entry.pooled_len == stock.pooled_len
    for mine, theirs in zip(entry.state, stock.state):
        assert _same(mine, theirs)
    x_after = _hidden(STEP, 11)
    with vision_rope(table, delta):
        assert _same(attn(x_after, entry), attn(x_after, stock))


def test_the_refused_shape_is_refused_because_it_really_differs(vision_attn):
    """The prompt ends less than one block after the image.

    The first block completed in decode starts on an image row, whose position
    is a grid position and not the sequence index plus the delta. The probe
    shows the mismatch and the admission rule names exactly this shape.
    """

    attn = vision_attn
    ids, table, delta, stock, bank = _prefilled(attn, 42, tail=1, seed=12)
    last_pad = max(i for i, token in enumerate(ids) if token == PAD)
    assert last_pad == (len(ids) // RATIO) * RATIO  # the block starts ON the image
    assert _admission(ids, table, delta)["refusal"] == "vision_tail_block_in_image"
    bank.rope_delta = delta
    x = _hidden(STEP, 13)
    with vision_rope(table, delta):
        golden = attn(x, stock)
    attn(x, bank)
    block = len(ids) // RATIO  # completed by this step
    assert not _same(bank.pooled[:, block], stock.pooled[:, block])
    # One token more of text and the block starts after the image: admitted,
    # and exact (covered above for every n % 4).
    ids_ok, table_ok, delta_ok = _image_prompt(43, tail=3)
    assert max(i for i, t in enumerate(ids_ok) if t == PAD) == 39
    assert _admission(ids_ok, table_ok, delta_ok)["refusal"] is None
    del golden


def test_without_the_delta_the_bank_is_wrong(vision_attn):
    """The guard this route replaces was right to exist."""

    attn = vision_attn
    _ids, table, delta, stock, bank = _prefilled(attn, 40, tail=8, seed=14)
    assert bank.rope_delta is None
    x = _hidden(STEP, 15)
    with vision_rope(table, delta):
        golden = attn(x, stock)
    assert not _same(attn(x, bank), golden)


def test_the_delta_moves_rotary_positions_and_nothing_else(vision_attn):
    """Offsets, values and raw indexer keys are KV-indexed: untouched."""

    attn = vision_attn
    _ids, _table, delta, _stock, with_delta = _prefilled(attn, 41, tail=9, seed=16)
    _ids, _table, _delta, _stock, without = _prefilled(attn, 41, tail=9, seed=16)
    with_delta.rope_delta = delta
    offset_before = with_delta.offset
    assert with_delta.rope_offset is not offset_before
    assert int(with_delta.rope_offset.item()) == 41 + delta
    assert with_delta.offset is offset_before  # reading the origin moves nothing
    for seed in (17, 18):
        x = _hidden(STEP, seed)
        attn(x, with_delta)
        attn(x, without)
    end = with_delta.size()
    assert end == without.size() == 41 + 2 * STEP
    # Written at the same rows, with the same unrotated content...
    assert _same(with_delta.kv.values[:, :, :end], without.kv.values[:, :, :end])
    assert _same(with_delta.raw_keys[:, :end], without.raw_keys[:, :end])
    # ...while everything rotary moved: keys, and the pooled keys of the
    # blocks completed in decode.
    assert not _same(with_delta.kv.keys[:, :, 41:end], without.kv.keys[:, :, 41:end])
    done = slice(41 // RATIO, end // RATIO)
    assert not _same(with_delta.pooled[:, done], without.pooled[:, done])
    assert _same(
        with_delta.pooled[:, : 41 // RATIO], without.pooled[:, : 41 // RATIO]
    )  # prompt blocks: prefilled, never re-rotated


def test_text_bank_rotary_origin_is_its_offset_object(attn):
    """The text lane must trace exactly the nodes it traced before."""

    fixed = [QSACache(compress_ratio=attn.indexer.ratio)]
    attn(_hidden(PREFILL, 19), fixed[0])
    bank = _promote(attn, fixed)
    assert bank.rope_delta is None
    assert bank.rope_offset is bank.offset
    attn(_hidden(STEP, 20), bank)
    assert bank.rope_offset is bank.offset  # still, after the offset advanced


def test_the_delta_is_one_int32_value():
    import mtplx.graphbank as gb

    delta = gb.as_rope_delta(-990)
    assert delta.dtype == mx.int32 and tuple(delta.shape) == (1,)
    assert int(delta.item()) == -990
    assert gb.as_rope_delta(None) is None
    scalar = gb.as_rope_delta(mx.array(7, dtype=mx.int32))
    assert tuple(scalar.shape) == (1,) and int(scalar.item()) == 7
    for bad in (
        mx.array([1], dtype=mx.int64),  # the rope kernels take int32 only
        mx.array([1.0]),
        mx.array([1, 2], dtype=mx.int32),
        1.5,
        True,
        "3",
    ):
        with pytest.raises(TypeError):
            gb.as_rope_delta(bad)
