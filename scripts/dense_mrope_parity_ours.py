#!/usr/bin/env python3
"""GPU cell: the MTPLX leg of a Prism parity run with image tokens at grid positions.

The Bonsai parity cell (``parity.py`` of the overnight cells, passed with
``--parity``) calls the loaded model directly, outside the generation loop, so
nothing arms the image position table there and its ``ours`` leg always ropes
image tokens sequentially. This script is that same leg with one difference:
the image forward runs inside the scope generation opens for a real image
request (``mtplx.dense_mrope``). It reuses the cell's own tokenizer, prompts,
synthetic image and result writer, and writes the same ``.npz`` + ``.json``
pair, so ``parity.py compare`` reads it unchanged.

What to look at in ``compare`` (image block of the new mode):
  trunk_native_after_image      should fall to the level trunk_sequential_after_image
                                had with sequential positions (KL mean about 6e-8,
                                top-1 24 of 24, max abs logit diff at float noise);
  trunk_sequential_after_image  rises to the old native gap: it is now the mismatched
                                pair. That swap is the proof.

``--mrope 0`` runs the leg with the table unarmed (the control, equal to the
cell's own ``ours`` leg). Loads ONE model. Never run on the CPU lane.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import sys
import time
from pathlib import Path


def _load_cell(path: Path):
    spec = importlib.util.spec_from_file_location("bonsai_parity_cell", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import the parity cell at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parity", required=True, help="path to the cell's parity.py")
    parser.add_argument("--pack", default=None, help="default: the cell's DEFAULT_PACK")
    parser.add_argument("--out", required=True, help="result .npz (a .json is written next to it)")
    parser.add_argument("--aux-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--share-rotation", type=int, choices=(0, 1), default=1)
    parser.add_argument("--mrope", type=int, choices=(0, 1), default=1)
    parser.add_argument("--min-free-gb", type=float, default=None)
    args = parser.parse_args()

    cell = _load_cell(Path(args.parity).resolve())
    pack = Path(args.pack) if args.pack else Path(cell.DEFAULT_PACK)
    cell.memory_guard(cell.MIN_FREE_GB if args.min_free_gb is None else args.min_free_gb)
    os.environ["MTPLX_PRISM_AUX_DTYPE"] = args.aux_dtype
    os.environ["MTPLX_PRISM_SHARE_ROTATION"] = "1" if args.share_rotation else "0"

    import mlx.core as mx
    import numpy as np

    from mtplx import demotions, runtime
    from mtplx.dense_mrope import INSTALL_ATTR, build_request_state, dense_mrope_scope
    from mtplx.vision import load_vision_tower, vision_spec_for_model_dir
    from mtplx.vision.splice import VisionSplice, spliced_chunk_embeddings

    prompts, image_ids, image_token_id = cell.tokenize(pack)
    started = time.time()
    rt = runtime.load(pack, mtp=False)
    load_s = time.time() - started
    model = rt.model
    install = getattr(model, INSTALL_ATTR, None)

    arrays: dict = {}
    dtypes: dict = {}
    for name, ids in prompts.items():
        logits = model(mx.array([ids]))
        mx.eval(logits)
        dtypes[name] = str(logits.dtype)
        arrays[f"text_{name}_ids"] = np.asarray(ids, dtype=np.int32)
        arrays[f"text_{name}_logits"] = np.asarray(logits[0].astype(mx.float32))

    spec = vision_spec_for_model_dir(pack)
    if spec is None or int(spec.image_token_id) != int(image_token_id):
        raise SystemExit("MTPLX does not see a vision tower in this pack")
    pixel_np, grid = cell.synthetic_pixel_values(pack)
    rows, _ = load_vision_tower(pack)(mx.array(pixel_np), [grid])
    mx.eval(rows)
    splice = VisionSplice(
        image_pad_token_id=int(image_token_id),
        embeddings=rows,
        image_digests=(0,),
        pad_counts=(int(rows.shape[0]),),
        image_grids=(grid,),
    )
    state = None
    if args.mrope:
        # The same call the serve layer makes for an image request.
        state = build_request_state(
            model,
            image_ids,
            image_token_id=int(image_token_id),
            image_grids=[grid],
            spatial_merge_size=int(spec.spatial_merge_size),
            video_token_id=int(spec.video_token_id),
        )
        if state is None:
            raise SystemExit(
                "the image position table was not armed: install record "
                f"{install!r}, demotions {demotions.snapshot()['reasons']!r}, "
                f"MTPLX_DENSE_MROPE={os.environ.get('MTPLX_DENSE_MROPE')!r}"
            )
    ids_mx = mx.array([image_ids])
    embedded = spliced_chunk_embeddings(
        model.language_model.model.embed_tokens, ids_mx, splice
    )
    with dense_mrope_scope(state) if state is not None else contextlib.nullcontext():
        logits = model(ids_mx, input_embeddings=embedded)
        mx.eval(logits)
    arrays["image_ids"] = np.asarray(image_ids, dtype=np.int32)
    arrays["image_token_id"] = np.asarray([image_token_id], dtype=np.int32)
    arrays["image_tower_rows"] = np.asarray(rows.astype(mx.float32))
    arrays["image_logits"] = np.asarray(logits[0].astype(mx.float32))
    dtypes["image"] = str(logits.dtype)

    cell.save_result(
        Path(args.out),
        arrays,
        {
            "engine": "mtplx",
            "mlx": mx.__version__,
            # ``compare`` labels a mode by this field; the suffix keeps the
            # grid-position run apart from the cell's own sequential run.
            "aux_dtype": f"{args.aux_dtype}+mrope" if state is not None else args.aux_dtype,
            "aux_dtype_loaded": args.aux_dtype,
            "share_rotation": bool(args.share_rotation),
            "dense_mrope": state is not None,
            "dense_mrope_delta": None if state is None else int(state.delta),
            "dense_mrope_install": None if install is None else repr(install),
            "demotions": demotions.snapshot()["counts"],
            "post_load_report": getattr(model, "_prism_post_load_report", None),
            "load_s": round(load_s, 2),
            "logit_dtypes": dtypes,
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        },
    )


if __name__ == "__main__":
    main()
