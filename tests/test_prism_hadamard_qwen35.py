"""prism_hadamard_qwen35 loader: math, fail-closed contract, vision, MTP graft.

Everything runs on the CPU device on a two-layer synthetic pack that has the
same container as Prism ML's Ternary Bonsai 2 27B pack (see
tests/prism_hadamard_synth.py). The reference in every comparison is a plain
dense model holding the weights the pack encodes, so a loader that skipped the
activation transform, or transformed something it should not, fails here.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.utils import load_model

from mtplx.models import prism_hadamard_qwen35 as ph
from tests import prism_hadamard_synth as synth

IDS = [[1, 5, 9, 200, 17, 33, 250, 3, 99, 120, 64, 12]]


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.delenv(ph.AUX_DTYPE_ENV, raising=False)
    monkeypatch.delenv(ph.SHARE_ROTATION_ENV, raising=False)
    yield
    mx.set_default_device(previous)


@pytest.fixture(scope="module")
def pack(tmp_path_factory) -> synth.SyntheticPack:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return synth.build_synthetic_pack(tmp_path_factory.mktemp("prism") / "pack")
    finally:
        mx.set_default_device(previous)


def _classes(config):
    return ph.Model, ph.ModelArgs


def _load(path: Path):
    model, _config = load_model(Path(path), get_model_classes=_classes)
    report = model.post_weight_load(path)
    return model, report


def _copy_pack(pack: synth.SyntheticPack, destination: Path) -> Path:
    shutil.copytree(pack.path, destination)
    return destination


def _rewrite_config(path: Path, mutate) -> None:
    config = json.loads((path / "config.json").read_text())
    mutate(config)
    (path / "config.json").write_text(json.dumps(config))


def _rewrite_tensors(path: Path, mutate) -> None:
    # mx.load is lazy: read everything before the file is replaced.
    tensors = dict(mx.load(str(path / "model.safetensors")))
    mutate(tensors)
    mx.eval(tensors)
    rewritten = path / "rewritten.safetensors"
    mx.save_safetensors(str(rewritten), tensors, metadata={"format": "mlx"})
    rewritten.replace(path / "model.safetensors")


def _logits(model, ids=IDS) -> np.ndarray:
    out = model(mx.array(ids))
    mx.eval(out)
    return np.asarray(out.astype(mx.float32))


# -- the transform -----------------------------------------------------------


def test_rotation_is_orthonormal_and_inverts():
    rng = np.random.default_rng(0)
    signs = mx.array(rng.choice([-1.0, 1.0], size=2048).astype(np.float32))
    a = mx.array(rng.standard_normal((3, 2048)).astype(np.float32))
    b = mx.array(rng.standard_normal((3, 2048)).astype(np.float32))
    ta = ph.hadamard_rotate(a, signs, 1024)
    tb = ph.hadamard_rotate(b, signs, 1024)
    back = ph.hadamard_rotate(ta, signs, 1024, inverse=True)
    assert float(mx.abs(back - a).max()) < 1e-5
    assert float(mx.abs((ta * tb).sum(-1) - (a * b).sum(-1)).max()) < 1e-3
    # Matches the float64 numpy definition T(x) = H(s * x).
    want = synth.rotate(np.asarray(a), np.asarray(signs))
    assert np.abs(np.asarray(ta) - want).max() < 1e-5


def test_rotation_keeps_the_input_dtype_and_survives_float16_outliers():
    signs = mx.ones((1024,), dtype=mx.float32)
    x = mx.full((1, 1024), 3000.0, dtype=mx.float16)
    # The unscaled butterfly would reach 3,072,000, far past float16's 65,504;
    # the float32 transform returns the exact normalized value 3000 * 32.
    y = ph.hadamard_rotate(x, signs, 1024)
    assert y.dtype == mx.float16
    assert float(y[0, 0]) == pytest.approx(3000.0 * 32.0, rel=1e-3) or bool(
        mx.isinf(y[0, 0])
    )
    assert float(mx.abs(y[0, 1:]).max()) == 0.0


def test_rotation_refuses_a_width_the_block_does_not_divide():
    with pytest.raises(ph.PrismHadamardContractError):
        ph.hadamard_rotate(mx.zeros((1, 1000)), mx.ones((1000,)), 1024)


# -- packed layers against the dense matrices they encode ----------------------


def _packed_linear(pack: synth.SyntheticPack, path: str) -> ph.HadamardQuantizedLinear:
    tensors = mx.load(str(pack.path / "model.safetensors"))
    key = "language_model." + path
    rows, width = pack.dense[path + ".weight"].shape
    layer = ph.HadamardQuantizedLinear(width, rows, block=synth.BLOCK)
    layer.load_weights(
        [(name, tensors[f"{key}.{name}"]) for name in ("weight", "scales", "biases", "signs")]
    )
    return layer


@pytest.mark.parametrize(
    "path",
    [
        "model.layers.1.self_attn.q_proj",
        "model.layers.0.linear_attn.out_proj",
        "model.layers.0.mlp.down_proj",
        "lm_head",
    ],
)
def test_rotated_quantized_linear_equals_the_dense_reference(pack, path):
    layer = _packed_linear(pack, path)
    dense = pack.dense[path + ".weight"].astype(np.float64)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 5, dense.shape[1])).astype(np.float32)
    want = x.astype(np.float64) @ dense.T
    scale = np.abs(want).max()

    got32 = np.asarray(layer(mx.array(x)))
    assert got32.dtype == np.float32
    assert np.abs(got32 - want).max() / scale < 2e-5

    got16 = layer(mx.array(x).astype(mx.float16))
    assert got16.dtype == mx.float16
    got16 = np.asarray(got16.astype(mx.float32))
    # float16 noise. The CPU quantized matmul accumulates each output
    # sequentially in float16 (about 1e-2 of the output scale at these widths);
    # the float32 check above is the proof of the math, this one covers the
    # dtype path: float16 in, float16 out, finite, and nowhere near garbage.
    assert np.isfinite(got16).all()
    assert np.abs(got16 - want).max() / scale < 3e-2

    # Without the activation transform the same packed weights give garbage.
    wrong = np.asarray(
        mx.quantized_matmul(
            mx.array(x),
            layer["weight"],
            scales=layer["scales"],
            biases=layer["biases"],
            transpose=True,
            group_size=128,
            bits=2,
        )
    )
    assert np.abs(wrong - want).max() / scale > 0.5


def test_rotated_embedding_returns_standard_basis_rows(pack):
    tensors = mx.load(str(pack.path / "model.safetensors"))
    key = "language_model.model.embed_tokens"
    embedding = ph.HadamardQuantizedEmbedding(synth.VOCAB, synth.HIDDEN, block=synth.BLOCK)
    embedding.load_weights(
        [(name, tensors[f"{key}.{name}"]) for name in ("weight", "scales", "biases", "signs")]
    )
    dense = pack.dense["model.embed_tokens.weight"]
    ids = mx.array([[0, 7, 383], [12, 12, 99]])
    rows = embedding(ids)
    assert rows.dtype == mx.float16 and rows.shape == (2, 3, synth.HIDDEN)
    want = dense[np.asarray(ids)]
    got = np.asarray(rows.astype(mx.float32))
    assert np.abs(got - want).max() / np.abs(want).max() < 2e-3
    # as_linear is the transpose use of the same table: <e, x> for every row.
    x = np.random.default_rng(5).standard_normal((1, 2, synth.HIDDEN)).astype(np.float32)
    got_linear = np.asarray(embedding.as_linear(mx.array(x)))
    want_linear = x.astype(np.float64) @ dense.astype(np.float64).T
    assert np.abs(got_linear - want_linear).max() / np.abs(want_linear).max() < 2e-5


def test_packed_layers_are_not_stock_quantized_layers():
    import mlx.nn as nn

    layer = ph.HadamardQuantizedLinear(1024, 8, block=1024)
    table = ph.HadamardQuantizedEmbedding(8, 1024, block=1024)
    # Fast paths that read packed tensors directly gate on these classes (and
    # on 4-bit or 8-bit); a rotated layer must never satisfy them.
    assert not isinstance(layer, (nn.QuantizedLinear, nn.Linear))
    assert not isinstance(table, (nn.QuantizedEmbedding, nn.Embedding))
    assert not hasattr(layer, "to_quantized")
    assert (layer.bits, layer.group_size) == (2, 128)


def test_shared_rotation_is_consumed_and_never_serves_another_input():
    shared = ph._SharedRotation(3)
    signs = mx.ones((1024,), dtype=mx.float32)
    x = mx.random.normal((1, 1024))
    first = shared(x, signs, 1024)
    assert shared(x, signs, 1024) is first
    assert shared(x, signs, 1024) is first
    assert shared._entry is None  # every sibling has read it
    other = mx.array(x)  # equal values, different array object
    assert shared(other, signs, 1024) is not first


# -- the whole pack ------------------------------------------------------------


def test_synthetic_pack_matches_the_dense_reference(pack, monkeypatch):
    reference = _logits(synth.build_dense_reference(pack))
    scale = np.abs(reference).max()

    model, report = _load(pack.path)
    assert report["packed_modules"] == len(pack.module_paths) == 15
    assert report["hadamard_config"] == "hadamard.json"
    assert report["shared_rotation_groups"] == 4  # qkv, gate/up twice, in_proj pair
    got = model(mx.array(IDS))
    assert got.dtype == mx.float16
    got = np.asarray(got.astype(mx.float32))
    assert (got.argmax(-1) == reference.argmax(-1)).all()
    assert np.abs(got - reference).max() / scale < 0.05

    # Reference precision (auxiliary tensors kept float32, as Prism's bundled
    # Python runtime effectively runs): the math itself is exact.
    monkeypatch.setenv(ph.AUX_DTYPE_ENV, "float32")
    model32, _ = _load(pack.path)
    got32 = _logits(model32)
    assert np.abs(got32 - reference).max() / scale < 2e-3


def test_shared_and_per_module_rotation_are_bit_identical(pack, monkeypatch):
    shared, report = _load(pack.path)
    monkeypatch.setenv(ph.SHARE_ROTATION_ENV, "0")
    separate, separate_report = _load(pack.path)
    assert report["shared_rotation_groups"] == 4
    assert separate_report["shared_rotation_groups"] == 0
    assert np.array_equal(_logits(shared), _logits(separate))


def _decode_windows(model, rows: int) -> list[np.ndarray]:
    """Prefill four tokens, then walk the rest of IDS in verify windows of ``rows``."""

    cache = model.make_cache()
    out = model(mx.array([IDS[0][:4]]), cache=cache)
    mx.eval(out)
    windows = []
    for start in range(4, len(IDS[0]) - rows + 1, rows):
        logits = model(mx.array([IDS[0][start : start + rows]]), cache=cache)
        mx.eval(logits)
        windows.append(np.asarray(logits.astype(mx.float32)))
    return windows


def test_every_packed_projection_is_proven_ternary(pack, monkeypatch):
    from mtplx.kernels import ternary_qmv

    monkeypatch.setattr(ternary_qmv, "MIN_N", 0)
    model, report = _load(pack.path)
    linears = [
        m for _r, m in model._packed_modules() if isinstance(m, ph.HadamardQuantizedLinear)
    ]
    assert linears and report["ternary_layout_modules"] == len(linears)
    assert all(m._ternary for m in linears)


def test_matrices_too_small_to_win_keep_the_stock_matmul(pack):
    from mtplx.kernels import ternary_qmv

    model, report = _load(pack.path)
    linears = [
        m for _r, m in model._packed_modules() if isinstance(m, ph.HadamardQuantizedLinear)
    ]
    big = [m for m in linears if int(m["weight"].shape[0]) >= ternary_qmv.MIN_N]
    assert big and len(big) < len(linears)
    assert report["ternary_layout_modules"] == len(big)
    big_ids = {id(m) for m in big}
    assert all(m._ternary == (id(m) in big_ids) for m in linears)


def test_a_non_ternary_matrix_keeps_the_stock_matmul(pack, tmp_path, monkeypatch):
    from mtplx.kernels import ternary_qmv

    monkeypatch.setattr(ternary_qmv, "MIN_N", 0)
    path = _copy_pack(pack, tmp_path / "pack")
    target = "language_model." + pack.module_paths[0] + ".biases"
    _rewrite_tensors(path, lambda t: t.__setitem__(target, t[target] * 0.5))
    model, report = _load(path)
    linears = [
        m for _r, m in model._packed_modules() if isinstance(m, ph.HadamardQuantizedLinear)
    ]
    assert report["ternary_layout_modules"] == len(linears) - 1


def test_fused_rotation_keeps_the_model_bit_identical(pack, monkeypatch):
    from mtplx.kernels import hadamard_rotate

    model, _ = _load(pack.path)
    monkeypatch.setenv(hadamard_rotate.ENV, "0")
    prefill_off, windows_off = _logits(model), _decode_windows(model, 2)
    monkeypatch.setenv(hadamard_rotate.ENV, "1")
    served = hadamard_rotate.counters()["served"]
    prefill_on, windows_on = _logits(model), _decode_windows(model, 2)
    assert hadamard_rotate.counters()["served"] > served
    assert np.array_equal(prefill_off, prefill_on)
    assert all(np.array_equal(a, b) for a, b in zip(windows_off, windows_on))


def _log_softmax64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_ternary_kernel_keeps_verify_windows_in_stocks_numerical_class(pack, monkeypatch, rows):
    """Same next-token law as stock at every window row, within float16 rounding.

    The kernel sums in a different float32 order than stock's qmv kernels, so
    logits are not bit-identical; on this random two-layer pack stock itself
    moves by up to about 2% of the logit range between its M = 1 and M = 2
    kernels. The bar is the distribution: the same argmax and a KL far below
    anything sampling can see.
    """
    from mtplx.kernels import ternary_qmv

    monkeypatch.setattr(ternary_qmv, "MIN_N", 0)
    model, _ = _load(pack.path)
    monkeypatch.setenv(ternary_qmv.ENV, "0")
    stock = _decode_windows(model, rows)
    monkeypatch.setenv(ternary_qmv.ENV, "1")
    served = ternary_qmv.counters()["served"]
    mine = _decode_windows(model, rows)
    assert ternary_qmv.counters()["served"] > served
    for a, b in zip(stock, mine):
        assert (a.argmax(-1) == b.argmax(-1)).all()
        la, lb = _log_softmax64(a), _log_softmax64(b)
        kl = (np.exp(la) * (la - lb)).sum(-1)
        assert float(kl.max()) < 1e-3
        assert float(np.abs(a - b).max()) / float(np.abs(a).max()) < 0.03


def test_float_policy_is_float16_with_no_bfloat16_anywhere(pack):
    from mlx.utils import tree_flatten

    model, report = _load(pack.path)
    assert report["float_dtypes"] == ["float16", "float32"]
    for name, value in tree_flatten(model.parameters()):
        assert value.dtype != mx.bfloat16, name
        if name.endswith(".signs"):
            assert value.dtype == mx.float32
        elif mx.issubdtype(value.dtype, mx.floating):
            assert value.dtype == mx.float16, name


def test_a_bfloat16_tensor_is_refused(pack, tmp_path):
    model, _ = _load(pack.path)
    norm = model.language_model.model.norm
    norm.weight = norm.weight.astype(mx.bfloat16)
    with pytest.raises(ph.PrismHadamardContractError, match="bfloat16"):
        model.post_weight_load(pack.path)


def test_stock_mlx_lm_cannot_load_the_pack(pack):
    from mlx_lm.utils import _get_classes

    with pytest.raises(ValueError):
        _get_classes(json.loads((pack.path / "config.json").read_text()))


def test_runtime_registers_the_model_type():
    from mtplx import runtime

    classes = runtime._model_classes_for_config({"model_type": "prism_hadamard_qwen35"})
    assert classes == (ph.Model, ph.ModelArgs)


# -- fail closed ---------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.pop("modules"), "modules"),
        (lambda c: c.update(modules=[]), "modules"),
        (lambda c: c.update(schema_version=1), "schema_version"),
        (lambda c: c.update(gdn_activation_layout="interleaved"), "gdn_activation_layout"),
        (lambda c: c.update(tensor_namespace="mlx-lm"), "tensor_namespace"),
        (lambda c: c.update(base_model_type="llama"), "base_model_type"),
        (lambda c: c.update(quantization={"bits": 4, "group_size": 64}), "quantization"),
        (lambda c: c["modules"][0].update(block=768), "block"),
        (lambda c: c["modules"][0].update(dtype="bfloat16"), "float16"),
        (lambda c: c["modules"].append(dict(c["modules"][0])), "duplicate"),
        (lambda c: c["modules"][0].update(path="model.layers.9.mlp.up_proj"), "does not exist"),
        (lambda c: c.pop("vision_config"), "vision"),
        (lambda c: c["components"].update(vision=False), "vision"),
    ],
)
def test_config_contract_violations_are_refused(pack, tmp_path, mutate, message):
    path = _copy_pack(pack, tmp_path / "pack")
    _rewrite_config(path, mutate)
    with pytest.raises(ph.PrismHadamardContractError, match=message):
        _load(path)


def test_a_module_missing_from_the_list_is_refused(pack, tmp_path):
    # An unlisted module would be built as a stock layer and run unrotated.
    path = _copy_pack(pack, tmp_path / "pack")
    _rewrite_config(
        path,
        lambda c: c.update(
            modules=[m for m in c["modules"] if m["path"] != "model.layers.0.mlp.up_proj"]
        ),
    )
    with pytest.raises(ph.PrismHadamardContractError, match="does not list"):
        _load(path)


def test_missing_sign_vector_is_refused(pack, tmp_path):
    path = _copy_pack(pack, tmp_path / "pack")
    _rewrite_tensors(
        path, lambda t: t.pop("language_model.model.layers.0.mlp.down_proj.signs")
    )
    with pytest.raises(ph.PrismHadamardContractError, match="signs"):
        _load(path)


def test_missing_rotation_metadata_file_is_refused(pack, tmp_path):
    path = _copy_pack(pack, tmp_path / "pack")
    (path / "hadamard.json").rename(path / "hadamard.json.aside")
    with pytest.raises(ph.PrismHadamardContractError, match="hadamard.json"):
        _load(path)


def test_sign_tensor_that_differs_from_the_metadata_is_refused(pack, tmp_path):
    path = _copy_pack(pack, tmp_path / "pack")

    def flip(tensors):
        key = "language_model.model.layers.1.self_attn.k_proj.signs"
        tensors[key] = tensors[key] * mx.concatenate(
            [mx.array([-1.0]), mx.ones((synth.HIDDEN - 1,))]
        )

    _rewrite_tensors(path, flip)
    with pytest.raises(ph.PrismHadamardContractError, match="differs"):
        _load(path)


def test_metadata_that_disagrees_with_the_module_list_is_refused(pack, tmp_path):
    path = _copy_pack(pack, tmp_path / "pack")
    meta = json.loads((path / "hadamard.json").read_text())
    meta["prism.hadamard.weight_names"] = meta["prism.hadamard.weight_names"][1:]
    (path / "hadamard.json").write_text(json.dumps(meta))
    with pytest.raises(ph.PrismHadamardContractError, match="weight_names"):
        _load(path)


def test_sign_values_are_checked_when_no_metadata_file_is_declared(pack, tmp_path):
    path = _copy_pack(pack, tmp_path / "pack")
    _rewrite_config(path, lambda c: c.pop("hadamard_config"))

    def corrupt(tensors):
        key = "language_model.lm_head.signs"
        tensors[key] = tensors[key] * 0.5

    _rewrite_tensors(path, corrupt)
    with pytest.raises(ph.PrismHadamardContractError, match=r"\+1 and -1"):
        _load(path)


# -- vision is mandatory -------------------------------------------------------


def test_pack_without_vision_tensors_is_refused(tmp_path):
    stripped = synth.build_synthetic_pack(tmp_path / "pack", with_vision=False)
    with pytest.raises(ph.PrismHadamardContractError, match="vision tower"):
        _load(stripped.path)


def test_pack_without_the_preprocessor_file_is_refused(tmp_path):
    stripped = synth.build_synthetic_pack(tmp_path / "pack", with_preprocessor=False)
    with pytest.raises(ph.PrismHadamardContractError, match="preprocessor_config.json"):
        _load(stripped.path)


def test_vision_spec_resolves_for_a_pack_without_a_weight_index(pack):
    from mtplx.vision import checkpoint_weight_map, vision_spec_for_model_dir

    assert not (pack.path / "model.safetensors.index.json").exists()
    weight_map = checkpoint_weight_map(pack.path)
    assert weight_map["vision_tower.merger.linear_fc2.weight"] == "model.safetensors"
    spec = vision_spec_for_model_dir(pack.path)
    assert spec is not None
    assert spec.image_token_id == synth.IMAGE_TOKEN_ID
    assert spec.out_hidden_size == synth.HIDDEN
    assert spec.model_type == "prism_hadamard_qwen35"


def _image_rows(pack: synth.SyntheticPack) -> tuple[mx.array, int]:
    from PIL import Image

    from mtplx.vision import load_vision_tower
    from mtplx.vision.processing import (
        decode_image,
        image_pad_token_count,
        preprocess_images,
    )

    pixels = np.random.default_rng(9).integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    preprocessor = json.loads((pack.path / "preprocessor_config.json").read_text())
    pixel_values, grids = preprocess_images([decode_image(buffer.getvalue())], preprocessor)
    tower = load_vision_tower(pack.path)
    rows, _deepstack = tower(pixel_values, grids)
    mx.eval(rows)
    return rows, image_pad_token_count(grids[0])


def test_vision_rows_enter_the_standard_basis_stream_untransformed(pack, monkeypatch):
    """Image rows replace embedding rows AFTER the inverse transform.

    The tower is stock and unrotated, the embedding lookup already returns
    standard-basis rows, and every projection rotates its own input. So the
    image rows need no transform: spliced as they are, the rotated model
    matches the dense reference; pushed through the embedding's inverse
    transform they do not.
    """

    from mtplx.vision.splice import VisionSplice, spliced_chunk_embeddings

    monkeypatch.setenv(ph.AUX_DTYPE_ENV, "float32")
    rows, pad_count = _image_rows(pack)
    assert rows.dtype == mx.float16  # the tower runs in its stored float16
    assert rows.shape == (pad_count, synth.HIDDEN)
    ids = mx.array([[1, synth.VISION_START_TOKEN_ID] + [synth.IMAGE_TOKEN_ID] * pad_count
                    + [synth.VISION_END_TOKEN_ID, 42, 7]])

    def run(model, image_rows):
        splice = VisionSplice(
            image_pad_token_id=synth.IMAGE_TOKEN_ID,
            embeddings=image_rows,
            image_digests=(1,),
            pad_counts=(pad_count,),
            image_grids=((1, 4, 4),),
        )
        embed = model.language_model.model.embed_tokens
        embedded = spliced_chunk_embeddings(embed, ids, splice)
        assert embedded is not None
        out = model(ids, input_embeddings=embedded)
        mx.eval(out)
        return np.asarray(out.astype(mx.float32))

    model, _ = _load(pack.path)
    reference = run(synth.build_dense_reference(pack), rows)
    scale = np.abs(reference).max()
    assert np.abs(run(model, rows) - reference).max() / scale < 2e-3

    embed = model.language_model.model.embed_tokens
    transformed = ph.hadamard_rotate(rows, embed["signs"], embed.block, inverse=True)
    assert np.abs(run(model, transformed) - reference).max() / scale > 0.05


# -- through mtplx.runtime, with the MTP graft -----------------------------------


def test_runtime_load_runs_autoregressive_and_with_a_grafted_mtp_head(pack, tmp_path):
    from mtplx import runtime

    path = _copy_pack(pack, tmp_path / "pack")
    reference = _logits(synth.build_dense_reference(pack))

    trunk_only = runtime.load(path, mtp=False)
    got = _logits(trunk_only.model)
    assert (got.argmax(-1) == reference.argmax(-1)).all()
    assert trunk_only.model._prism_post_load_report["packed_modules"] == 15

    synth.write_synthetic_mtp_sidecar(synth.SyntheticPack(path, {}, {}, {}, []))
    grafted = runtime.load(path, mtp=True)
    assert grafted.mtp_enabled
    ids = mx.array(IDS)
    logits, hidden = grafted.model(ids, return_hidden=True)
    mx.eval(logits, hidden)
    # The trunk is untouched by the graft, and the hidden state handed to the
    # head is the standard-basis post-norm stream of a stock Qwen3.5 model.
    assert np.array_equal(np.asarray(logits.astype(mx.float32)), got)
    reference_model = synth.build_dense_reference(pack)
    want_hidden = reference_model.language_model.model(ids)
    mx.eval(want_hidden)
    want_hidden = np.asarray(want_hidden)
    got_hidden = np.asarray(hidden.astype(mx.float32))
    assert np.abs(got_hidden - want_hidden).max() / np.abs(want_hidden).max() < 0.05

    draft_logits = grafted.model.mtp_forward(
        hidden[:, :-1, :], ids[:, 1:], mtp_cache=grafted.model.make_mtp_cache()
    )
    mx.eval(draft_logits)
    assert draft_logits.shape == (1, len(IDS[0]) - 1, synth.VOCAB)
    assert draft_logits.dtype == mx.float16
    assert bool(mx.all(mx.isfinite(draft_logits)))
