"""Flash-Next (qwen4_exp) vision M-RoPE: table math, axis layout, attention.

The shipped packs carry mrope_section [11, 11, 10] with mrope_interleaved
true; before this work the fields were silently dropped and image tokens got
plain 1-D rope — numerically wrong positions for a vision request. These
tests pin the reference semantics (mlx-vlm / transformers Qwen-VL family):

- interleaved axis layout t@0,3..30 / h@1,4..31 / w@2,5..29
- position contraction (an image occupies max(t, h/2, w/2) positions)
- equal-axes tables reduce bit-exactly to plain rope (text safety)
- vision retains QSA selection, with the reference M-RoPE positions applied
  to indexer queries and the first token of each pooled-key block
"""

import mlx.core as mx
import numpy as np
import pytest

from mtplx.attention_context import vision_rope, vision_rope_state
from mtplx.models.qwen4_exp import (
    QSACache,
    QSAIndexer,
    TextArgs,
    _build_mrope_axes,
    _mrope_cos_sin,
    _rope_cos_sin,
)
from mtplx.vision.mrope import build_mrope_positions


def test_interleaved_axis_layout_matches_reference():
    axes = _build_mrope_axes([11, 11, 10], interleaved=True)
    assert len(axes) == 32
    assert [i for i, a in enumerate(axes) if a == 0] == list(range(0, 31, 3))
    assert [i for i, a in enumerate(axes) if a == 1] == list(range(1, 32, 3))
    assert [i for i, a in enumerate(axes) if a == 2] == list(range(2, 30, 3))


def test_contiguous_axis_layout():
    assert _build_mrope_axes([2, 2, 1], interleaved=False) == [0, 0, 1, 1, 2]


def test_equal_axes_reduce_to_plain_rope_bit_exactly():
    # Text tokens carry equal (t, h, w); the mrope tables must then be
    # bit-identical to the plain rope tables — the text-safety invariant.
    inv_freq = 1e7 ** (-mx.arange(0, 16, 2, dtype=mx.float32) / 16)
    axes = mx.array(_build_mrope_axes([3, 3, 2], True), dtype=mx.int32)
    positions = mx.arange(7, 19, dtype=mx.int32)
    table = mx.broadcast_to(positions[None, :], (3, 12))
    cos_m, sin_m = _mrope_cos_sin(table, inv_freq, axes)
    cos_p, sin_p = _rope_cos_sin(positions, inv_freq)
    assert mx.array_equal(cos_m, cos_p).item()
    assert mx.array_equal(sin_m, sin_p).item()


def test_build_positions_single_image_contraction_and_delta():
    # [text text | 2x2-llm image (4 pads) | text]; grid (1, 4, 4), merge 2.
    ids = [1, 2, 99, 99, 99, 99, 3]
    built = build_mrope_positions(
        ids, image_token_id=99, image_grids=[(1, 4, 4)], spatial_merge_size=2
    )
    assert built is not None
    table, delta = built
    assert table[0].tolist() == [0, 1, 2, 2, 2, 2, 4]
    assert table[1].tolist() == [0, 1, 2, 2, 3, 3, 4]
    assert table[2].tolist() == [0, 1, 2, 3, 2, 3, 4]
    assert delta == -2  # positions end at 4+1=5 for 7 tokens


def test_build_positions_multi_image_and_refusals():
    ids = [7, 99, 99, 99, 99, 8, 99, 99, 99, 99]
    built = build_mrope_positions(
        ids,
        image_token_id=99,
        image_grids=[(1, 4, 4), (1, 4, 4)],
        spatial_merge_size=2,
    )
    assert built is not None
    table, delta = built
    # text@0, img1@1..2, text@3, img2@4..5 -> max 5, delta = 6 - 10 = -4
    assert table[0].tolist() == [0, 1, 1, 1, 1, 3, 4, 4, 4, 4]
    assert delta == -4
    # More pads than grids: refuse rather than mis-rope.
    assert (
        build_mrope_positions(
            ids, image_token_id=99, image_grids=[(1, 4, 4)], spatial_merge_size=2
        )
        is None
    )
    # Video pads: refuse.
    assert (
        build_mrope_positions(
            [1, 55],
            image_token_id=99,
            image_grids=[],
            spatial_merge_size=2,
            video_token_id=55,
        )
        is None
    )


def _tiny_attention():
    from mtplx.models.qwen4_exp import Attention

    args = TextArgs.from_dict(
        {
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 512,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [2, 1, 1],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "indexer_n_heads": 0,
        }
    )
    return Attention(args)


def test_attention_equal_axes_table_matches_plain_path():
    attn = _tiny_attention()
    assert attn._mrope_axes is not None
    x = mx.random.normal((1, 6, 128)).astype(mx.bfloat16)

    plain = attn(x, QSACache(4))
    table = mx.broadcast_to(mx.arange(6, dtype=mx.int32)[None, :], (3, 6))
    with vision_rope(table, 0):
        vision = attn(x, QSACache(4))
    assert mx.array_equal(plain, vision).item()


def test_attention_delta_branch_shifts_positions():
    attn = _tiny_attention()
    x = mx.random.normal((1, 4, 128)).astype(mx.bfloat16)
    # Past the table (None table, delta d) the rope positions are
    # sequence_index + d. The rotated keys a forward stores are the witness:
    # rows written at index 0..3 under delta d must be, bit for bit, the rows
    # the plain path writes at index d..d+3 (same projections of the same
    # rows, rotated at the same absolute positions).
    delta = 5
    shifted_cache = QSACache(4)
    with vision_rope(None, delta):
        attn(x, shifted_cache)
    plain_cache = QSACache(4)
    attn(mx.random.normal((1, delta, 128)).astype(mx.bfloat16), plain_cache)
    attn(x, plain_cache)
    shifted_keys = shifted_cache.kv.keys[:, :, :4]
    assert mx.array_equal(
        shifted_keys, plain_cache.kv.keys[:, :, delta : delta + 4]
    ).item()
    # And the delta is really applied: without it the same rows rotate at
    # 0..3 and come out different.
    unshifted_cache = QSACache(4)
    attn(x, unshifted_cache)
    assert not mx.array_equal(shifted_keys, unshifted_cache.kv.keys[:, :, :4]).item()
    # Values carry no position, so they are the same rows in all three runs.
    assert mx.array_equal(
        shifted_cache.kv.values[:, :, :4], unshifted_cache.kv.values[:, :, :4]
    ).item()
    assert mx.array_equal(
        shifted_cache.kv.values[:, :, :4], plain_cache.kv.values[:, :, delta : delta + 4]
    ).item()


def _single_image_table():
    # [text text | 2x2-llm image (4 pads) | text]: 7 rows, delta -2. Rows 0..5
    # sit at positions that are NOT index + delta; row 6 is the first that is.
    ids = [1, 2, 99, 99, 99, 99, 3]
    table, delta = build_mrope_positions(
        ids, image_token_id=99, image_grids=[(1, 4, 4)], spatial_merge_size=2
    )
    return mx.array(table), int(delta)


@pytest.mark.parametrize(
    ("start", "rows"),
    [
        (0, 7),  # the whole table
        (2, 4),  # inside the table: the image rows
        (7, 4),  # past the table
        (9, 3),  # past the table, further on
        (0, 11),  # straddles the end: every table row plus four rows past it
        (4, 6),  # straddles the end from inside the image
        (5, 3),  # straddles the end with one image row on the inside
        (6, 2),  # straddles the end at the one row both rules agree on
    ],
)
def test_attention_ropes_every_row_of_a_chunk_by_its_own_rule(start, rows):
    """One forward may hold rows inside the table and rows past it.

    A re-prefill of prompt plus generated tokens (MTPLX_STATE_REBASE_EVERY) is
    such a forward. The rule is per row: a row inside the table rotates at its
    (t, h, w) table position, a row past it at index + delta. The witness is
    the rotated keys the forward stores, against an independent spelling of
    the positions (one [3, S] table built row by row).
    """

    attn = _tiny_attention()
    table, delta = _single_image_table()
    table_len = int(table.shape[1])
    x = mx.random.normal((1, start + rows, 128)).astype(mx.bfloat16)
    chunk = x[:, start:]

    cache = QSACache(4)
    with vision_rope(table, delta):
        if start:
            # The rows before the window, in forwards that do not straddle.
            attn(x[:, : min(start, table_len)], cache)
            if start > table_len:
                attn(x[:, table_len:start], cache)
        attn(chunk, cache)

    per_row = [
        [int(v) for v in table[:, i].tolist()] if i < table_len else [i + delta] * 3
        for i in range(start, start + rows)
    ]
    positions3 = mx.array(per_row, dtype=mx.int32).T
    cos, sin = _mrope_cos_sin(positions3, attn._inv_freq, attn._mrope_axes)
    from mtplx.models.qwen4_exp import _apply_partial_rope

    keys = attn.k_norm(attn.k_proj(chunk).reshape(1, rows, attn.n_kv_heads, -1))
    expected = _apply_partial_rope(keys, cos, sin).transpose(0, 2, 1, 3)
    stored = cache.kv.keys[:, :, start : start + rows]
    assert mx.array_equal(stored, expected).item()


def test_a_straddling_chunk_stores_the_keys_of_the_same_rows_fed_in_two_chunks():
    attn = _tiny_attention()
    table, delta = _single_image_table()
    x = mx.random.normal((1, 11, 128)).astype(mx.bfloat16)
    with vision_rope(table, delta):
        split = QSACache(4)
        attn(x[:, :7], split)  # wholly inside the table
        attn(x[:, 7:], split)  # wholly past it
        straddling = QSACache(4)
        attn(x, straddling)  # one forward over both
    assert mx.array_equal(
        straddling.kv.keys[:, :, :11], split.kv.keys[:, :, :11]
    ).item()
    assert mx.array_equal(
        straddling.kv.values[:, :, :11], split.kv.values[:, :, :11]
    ).item()


def test_chunk_tables_off_the_straddle_are_the_tables_built_before():
    from mtplx.models.qwen4_exp import _vision_chunk_cos_sin

    table, delta = _single_image_table()
    inv_freq = 1e7 ** (-mx.arange(0, 8, 2, dtype=mx.float32) / 8)
    axes = mx.array(_build_mrope_axes([2, 1, 1], True), dtype=mx.int32)

    def same(actual, expected):
        return all(mx.array_equal(a, e).item() for a, e in zip(actual, expected))

    # Wholly inside the table: the table slice, as before.
    assert same(
        _vision_chunk_cos_sin(table, delta, 2, 4, inv_freq, axes),
        _mrope_cos_sin(table[:, 2:6], inv_freq, axes),
    )
    # Wholly past it, or no table at all (the decode scope): plain rope at
    # index + delta, as before.
    shifted = _rope_cos_sin(mx.arange(7 + delta, 11 + delta, dtype=mx.int32), inv_freq)
    assert same(_vision_chunk_cos_sin(table, delta, 7, 4, inv_freq, axes), shifted)
    assert same(_vision_chunk_cos_sin(None, delta, 7, 4, inv_freq, axes), shifted)
    # Straddling: the rows of the first followed by the rows of the second.
    cos, sin = _vision_chunk_cos_sin(table, delta, 2, 9, inv_freq, axes)
    inside = _mrope_cos_sin(table[:, 2:7], inv_freq, axes)
    assert same((cos[:5], sin[:5]), inside) and same((cos[5:], sin[5:]), shifted)
    # A scaled rope type scales every row, the table rows included.
    scaled = _vision_chunk_cos_sin(table, delta, 2, 9, inv_freq, axes, 1.25)
    assert same(scaled, (cos * 1.25, sin * 1.25))


def test_attention_vision_scope_preserves_sparse_selection():
    attn = _tiny_attention()

    class _PoisonIndexer:
        def __call__(self, x, pos_start, cache, qk_rows=None):
            # The official model always intersects attention with this mask,
            # including vision. Dense attention would change the model.
            S = x.shape[1]
            eye = mx.eye(S, dtype=mx.bool_)[None, None]
            return eye

    x = mx.random.normal((1, 5, 128)).astype(mx.bfloat16)
    dense = attn(x, QSACache(4))

    attn.indexer = _PoisonIndexer()
    poisoned = attn(x, QSACache(4))
    assert not mx.array_equal(dense, poisoned).item()

    table = mx.broadcast_to(mx.arange(5, dtype=mx.int32)[None, :], (3, 5))
    with vision_rope(table, 0):
        vision = attn(x, QSACache(4))
    assert mx.array_equal(poisoned, vision).item()


def test_vision_indexer_query_and_block_start_positions_match_numpy():
    args = TextArgs(hidden_size=32, head_dim=32, indexer_head_dim=32,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_compress_ratio=2,
        rope_parameters={"mrope_interleaved": True, "mrope_section": [2, 1, 1],
                         "partial_rotary_factor": .25, "rope_theta": 10000000,
                         "rope_type": "default"})
    indexer = QSAIndexer(args)
    rng = np.random.default_rng(607)
    raw_q = rng.standard_normal((1, 10, 2, 32)).astype(np.float32)
    raw_k = rng.standard_normal((1, 14, 32)).astype(np.float32)
    table = np.array([[0,1,2,3,4,4,4,4,6,7,8,9],
                      [0,1,2,3,4,4,5,5,6,7,8,9],
                      [0,1,2,3,4,5,4,5,6,7,8,9]], dtype=np.int32)
    inv = np.asarray(indexer._inv_freq)

    def oracle(values, positions):
        # Independent reference arithmetic: RMSNorm, then rotate at each
        # query / pooled-block FIRST position, with equal axes after images.
        values = values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + args.rms_norm_eps)
        pos3 = np.stack([table[:, p] if p < table.shape[1] else np.full(3, p-2)
                         for p in positions], axis=1)
        angles = pos3[[0,1,2,0]].T.astype(np.float32) * inv[None, :]
        angles = np.concatenate([angles, angles], axis=-1)[None, :, None, :]
        first = values[..., :8]
        rotated = np.concatenate([-first[..., 4:], first[..., :4]], axis=-1)
        return np.concatenate([first*np.cos(angles) + rotated*np.sin(angles), values[..., 8:]], axis=-1)

    with vision_rope(mx.array(table), -2):
        actual_q = indexer._prepare_queries(mx.array(raw_q), 4)
        actual_k = indexer._pool_keys_eager(mx.array(raw_k), 0, 7)
    expected_q = oracle(raw_q, np.arange(4, 14))
    expected_k = oracle(raw_k.reshape(1,7,2,32).mean(axis=2)[:, :, None], np.arange(0,14,2))[:, :, 0]
    np.testing.assert_allclose(np.asarray(actual_q), expected_q, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual_k), expected_k, rtol=2e-6, atol=2e-6)


def test_vision_rope_scope_helper_and_wiring():
    import inspect

    import mtplx.generation as generation

    # Prompt-state builder is wrapped (covers request, warm-restore and
    # postcommit prefill forwards)...
    assert hasattr(generation.restore_or_prefill_prompt_state, "__wrapped__")
    # ...and EVERY trunk forward of the decode loop arms the scope through one
    # helper, never by hand at a single site: through 2.11.3 only the main
    # verify did, and copy rounds, repairs and the final commit roped at the
    # raw index (tests/test_vision_trunk_forward_positions.py holds the
    # call-site proof and the bit-for-bit rows).
    src = inspect.getsource(generation.generate_mtpk)
    assert src.count("_decode_trunk_scope(vision_splice)") >= 8
    assert "_vision_rope_scope_for(vision_splice)" not in src
    assert 'attention_phase("decode_verify")' not in src
    # Helper: nullcontext for text, armed scope for a vision splice.
    import contextlib

    assert isinstance(
        generation._vision_rope_scope_for(None), contextlib.nullcontext
    )

    class _S:
        mrope_table = None
        mrope_delta = -3

    with generation._vision_rope_scope_for(_S()):
        assert vision_rope_state() == (None, -3)
    assert vision_rope_state() is None
