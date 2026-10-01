#!/usr/bin/env python3
"""Cross-check the dense-path image rope against mlx-vlm's qwen3_5 code.

Not a unit test: it needs an interpreter that has ``mlx_vlm`` installed (the
product venv does not), with this checkout on PYTHONPATH:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<checkout> <venv with mlx-vlm>/bin/python \
        scripts/dense_mrope_reference_check.py

Tiny random tensors only, no model and no pack. Two comparisons, both against
the reference implementation itself rather than a re-statement of it:

  positions  mtplx.vision.mrope.build_mrope_positions against
             mlx_vlm qwen3_5 LanguageModel.get_rope_index (table and delta),
             for one image, two images, and an image at the very start;
  rotation   mtplx.dense_mrope.DenseMRopeAdapter against
             mlx_vlm Qwen3_5RotaryEmbedding.apply_rotary on the same queries
             and keys at the pack layout ([11, 11, 10] over 64 of 256 dims,
             theta 1e7), whole and in cache-offset chunks, plus rows after the
             prompt at index + delta.

Prints one JSON line; exit 1 when a comparison is out of tolerance.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SECTION = [11, 11, 10]
ROTARY_DIMS = 64
HEAD_DIM = 256
THETA = 10_000_000.0
PAD, VIDEO, START, END = 900, 901, 902, 903
MERGE = 2
# mx.fast.rope uses fast trig, the reference exact trig: float32 agreement.
TOLERANCE = 2e-4


def _prompt(images: list[tuple[int, int, int]], lead: int, gap: int, tail: int) -> list[int]:
    ids = list(range(1, 1 + lead))
    for t, h, w in images:
        ids += [START] + [PAD] * (t * (h // MERGE) * (w // MERGE)) + [END]
        ids += list(range(100, 100 + gap))
    return ids + list(range(200, 200 + tail))


def _reference_positions(ids: list[int], grids: list[tuple[int, int, int]]):
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    stub = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=MERGE),
            image_token_id=PAD,
            video_token_id=VIDEO,
            vision_start_token_id=START,
        )
    )
    position_ids, deltas = LanguageModel.get_rope_index(
        stub, mx.array([ids]), image_grid_thw=mx.array([list(g) for g in grids])
    )
    return np.asarray(position_ids)[:, 0, :], int(np.asarray(deltas).reshape(-1)[0])


def main() -> int:
    from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

    from mtplx.dense_mrope import (
        ROLE_TRUNK,
        DenseMRopeAdapter,
        DenseMRopeState,
        dense_mrope_scope,
        mrope_axes,
    )
    from mtplx.vision.mrope import build_mrope_positions

    report: dict = {"positions": [], "rotation": {}}
    ok = True

    cases = {
        "one_image": ([(1, 28, 28)], 7, 0, 24),
        "two_images": ([(1, 8, 12), (1, 20, 6)], 5, 9, 11),
        "image_first": ([(1, 6, 4)], 0, 0, 5),
    }
    tables = {}
    for name, (grids, lead, gap, tail) in cases.items():
        ids = _prompt(grids, lead, gap, tail)
        want_table, want_delta = _reference_positions(ids, grids)
        table, delta = build_mrope_positions(
            ids, image_token_id=PAD, image_grids=grids, spatial_merge_size=MERGE,
            video_token_id=VIDEO,
        )
        same = bool(np.array_equal(table, want_table) and delta == want_delta)
        ok &= same
        tables[name] = (ids, table, delta)
        report["positions"].append(
            {"case": name, "tokens": len(ids), "delta": delta, "reference_delta": want_delta, "equal": same}
        )

    reference = Qwen3_5RotaryEmbedding(
        ROTARY_DIMS, max_position_embeddings=4096, base=THETA, mrope_section=SECTION
    )
    adapter = DenseMRopeAdapter(
        nn.RoPE(ROTARY_DIMS, traditional=False, base=THETA),
        mrope_axes(SECTION, True, ROTARY_DIMS // 2),
        ROLE_TRUNK,
    )
    rng = np.random.default_rng(3)
    for name, (ids, table, delta) in tables.items():
        n, extra = len(ids), 6
        q = mx.array(rng.standard_normal((1, 4, n + extra, HEAD_DIM)).astype(np.float32))
        k = mx.array(rng.standard_normal((1, 2, n + extra, HEAD_DIM)).astype(np.float32))
        tail = np.arange(n, n + extra, dtype=np.int32) + delta
        positions = np.concatenate([table, np.broadcast_to(tail, (3, extra))], axis=1)
        want_q, want_k = reference.apply_rotary(
            q, k, mx.array(positions)[:, None, :], unsqueeze_dim=1
        )
        state = DenseMRopeState(table, delta)
        cut = max(1, n // 3)
        with dense_mrope_scope(state):
            got_q = adapter(q, offset=0)
            got_k = mx.concatenate(
                [
                    adapter(k[:, :, :cut], offset=0),
                    adapter(k[:, :, cut:n], offset=cut),
                    adapter(k[:, :, n:], offset=n),  # rows after the prompt
                ],
                axis=2,
            )
        gap_q = float(mx.abs(got_q - want_q).max().item())
        gap_k = float(mx.abs(got_k - want_k).max().item())
        sequential = float(mx.abs(adapter.inner(q, offset=0) - want_q).max().item())
        ok &= gap_q < TOLERANCE and gap_k < TOLERANCE and sequential > 100 * TOLERANCE
        report["rotation"][name] = {
            "max_abs_diff_queries": gap_q,
            "max_abs_diff_keys_chunked": gap_k,
            "sequential_rope_would_differ_by": sequential,
        }

    report["ok"] = bool(ok)
    report["mlx"] = mx.__version__
    print(json.dumps(report, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
