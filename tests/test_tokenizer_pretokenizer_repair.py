"""The loader restores the Qwen3 pre-tokenizer regex (letters + combining marks
stay in one word) when transformers or a re-saved pack split them apart."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Regex, Tokenizer, models, pre_tokenizers

from mtplx.runtime import (
    _LEGACY_QWEN2_PRETOKENIZER_SPLIT,
    _QWEN3_PRETOKENIZER_SPLIT,
    _pretokenizer_split_pattern,
    restore_qwen3_pretokenizer,
)

HINDI = "भारत"  # letters + combining marks (matra)


def _byte_level_tokenizer(pattern: str) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
        ]
    )
    return tok


class _FastLike:
    """Minimal stand-in for a HF fast tokenizer (has backend_tokenizer)."""

    def __init__(self, backend: Tokenizer) -> None:
        self.backend_tokenizer = backend


def _write_pack(tmp_path: Path, file_pattern: str, model_type: str) -> Path:
    tok = _byte_level_tokenizer(file_pattern)
    (tmp_path / "tokenizer.json").write_text(tok.to_str(), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return tmp_path


def _pieces(backend: Tokenizer, text: str) -> int:
    return len(backend.pre_tokenizer.pre_tokenize_str(text))


def test_the_two_regexes_split_devanagari_differently():
    legacy = _byte_level_tokenizer(_LEGACY_QWEN2_PRETOKENIZER_SPLIT)
    qwen3 = _byte_level_tokenizer(_QWEN3_PRETOKENIZER_SPLIT)
    assert _pieces(qwen3, HINDI) == 1
    assert _pieces(legacy, HINDI) > 1
    assert _pieces(legacy, "hello world") == _pieces(qwen3, "hello world")


def test_loaded_backend_follows_the_file(tmp_path):
    # transformers' Qwen2Tokenizer rebuilt the backend with its own regex
    pack = _write_pack(tmp_path, _QWEN3_PRETOKENIZER_SPLIT, "qwen4_exp")
    loaded = _FastLike(_byte_level_tokenizer(_LEGACY_QWEN2_PRETOKENIZER_SPLIT))
    receipt = restore_qwen3_pretokenizer(loaded, pack, {"model_type": "qwen4_exp"})
    assert receipt is not None and receipt["source"] == "tokenizer.json"
    assert _pieces(loaded.backend_tokenizer, HINDI) == 1


def test_qwen3_family_pack_with_the_qwen2_regex_baked_in_is_repaired(tmp_path):
    # the 27B packs were re-saved through Qwen2Tokenizer and ship the old regex
    pack = _write_pack(tmp_path, _LEGACY_QWEN2_PRETOKENIZER_SPLIT, "qwen3_5")
    loaded = _FastLike(_byte_level_tokenizer(_LEGACY_QWEN2_PRETOKENIZER_SPLIT))
    receipt = restore_qwen3_pretokenizer(loaded, pack, {"model_type": "qwen3_5"})
    assert receipt is not None and "qwen3_5" in receipt["source"]
    loaded_pattern = _pretokenizer_split_pattern(
        json.loads(bytes(loaded.backend_tokenizer.pre_tokenizer.__getstate__()).decode())
    )
    assert loaded_pattern == _QWEN3_PRETOKENIZER_SPLIT
    assert _pieces(loaded.backend_tokenizer, HINDI) == 1


def test_a_foreign_family_with_the_qwen2_regex_is_left_alone(tmp_path):
    pack = _write_pack(tmp_path, _LEGACY_QWEN2_PRETOKENIZER_SPLIT, "qwen2")
    loaded = _FastLike(_byte_level_tokenizer(_LEGACY_QWEN2_PRETOKENIZER_SPLIT))
    assert restore_qwen3_pretokenizer(loaded, pack, {"model_type": "qwen2"}) is None


def test_a_correct_load_is_a_no_op(tmp_path):
    pack = _write_pack(tmp_path, _QWEN3_PRETOKENIZER_SPLIT, "qwen3_5")
    loaded = _FastLike(_byte_level_tokenizer(_QWEN3_PRETOKENIZER_SPLIT))
    assert restore_qwen3_pretokenizer(loaded, pack, {"model_type": "qwen3_5"}) is None


_OQ = Path.home() / ".mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Quality"


@pytest.mark.skipif(not (_OQ / "tokenizer.json").exists(), reason="pack not installed")
def test_the_shipped_27b_quality_pack_tokenizes_like_upstream():
    from mtplx.runtime import _load_tokenizer_resilient

    tok = _load_tokenizer_resilient(_OQ, json.loads((_OQ / "config.json").read_text()))
    counts = {
        "hindi": len(tok.encode("भारत एक विशाल देश है जिसकी संस्कृति बहुत पुरानी है।", add_special_tokens=False)),
        "thai": len(tok.encode("ประเทศไทยมีวัฒนธรรมที่เก่าแก่และสวยงามมาก", add_special_tokens=False)),
        "english": len(tok.encode("The quick brown fox jumps over the lazy dog near the river bank.", add_special_tokens=False)),
    }
    # upstream Qwen/Qwen3.8-27B tokenizer.json counts (2026-09-08 receipt)
    assert counts == {"hindi": 20, "thai": 9, "english": 14}
