"""Content-keyed session caching for vision prompts.

The session bank keys on token-id prefixes, and every image shares one pad
token id — so raw vision prompts were banned from the bank outright (blanket
bypass, full re-prefill every follow-up turn). These tests pin the surrogate
keying that lifts the ban: pad positions remapped to ids derived from each
image's content digest, making the key sequence a pure function of
(text tokens, pixels, positions).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.vision.splice import (
    _BANK_KEY_FLAG,
    VisionSplice,
    vision_bank_key_ids,
)

PAD = 999  # stand-in image pad token id


def make_splice(digests, pad_counts, total_rows=None):
    rows = total_rows if total_rows is not None else sum(pad_counts)
    return VisionSplice(
        image_pad_token_id=PAD,
        embeddings=mx.zeros((rows, 8)),
        image_digests=tuple(digests),
        pad_counts=tuple(pad_counts),
    )


class TestVisionBankKeyIds:
    def test_text_positions_untouched(self):
        prompt = [1, 2, PAD, PAD, 3, 4]
        keyed = vision_bank_key_ids(prompt, make_splice([0xABCD], [2]))
        assert keyed is not None
        assert keyed[0:2] == [1, 2]
        assert keyed[4:6] == [3, 4]

    def test_pad_positions_remapped_out_of_vocab(self):
        prompt = [1, PAD, PAD, 2]
        keyed = vision_bank_key_ids(prompt, make_splice([0xABCD], [2]))
        assert keyed is not None
        for pos in (1, 2):
            assert keyed[pos] != PAD
            assert keyed[pos] & _BANK_KEY_FLAG

    def test_same_image_same_keys(self):
        prompt = [1, PAD, PAD, 2]
        splice_a = make_splice([0xABCD], [2])
        splice_b = make_splice([0xABCD], [2])
        assert vision_bank_key_ids(prompt, splice_a) == vision_bank_key_ids(
            prompt, splice_b
        )

    def test_different_image_different_keys(self):
        prompt = [1, PAD, PAD, 2]
        keyed_a = vision_bank_key_ids(prompt, make_splice([0xABCD], [2]))
        keyed_b = vision_bank_key_ids(prompt, make_splice([0xEF01], [2]))
        assert keyed_a is not None and keyed_b is not None
        assert keyed_a != keyed_b
        # Divergence is exactly at the pad positions.
        assert keyed_a[0] == keyed_b[0]
        assert keyed_a[3] == keyed_b[3]
        assert keyed_a[1] != keyed_b[1]
        assert keyed_a[2] != keyed_b[2]

    def test_rows_within_one_image_are_distinct(self):
        prompt = [PAD, PAD, PAD]
        keyed = vision_bank_key_ids(prompt, make_splice([0xABCD], [3]))
        assert keyed is not None
        assert len(set(keyed)) == 3

    def test_multi_image_ordering(self):
        prompt = [1, PAD, 2, PAD, PAD, 3]
        keyed_ab = vision_bank_key_ids(prompt, make_splice([0xA, 0xB], [1, 2]))
        keyed_ba = vision_bank_key_ids(prompt, make_splice([0xB, 0xA], [1, 2]))
        assert keyed_ab is not None and keyed_ba is not None
        # Swapping which image sits where must change the key sequence.
        assert keyed_ab != keyed_ba

    def test_prefix_stability_for_appended_turns(self):
        # An OpenCode follow-up strictly extends the prompt; the keyed view
        # of the shared prefix must be byte-identical or warm restores break.
        splice = make_splice([0xABCD], [2])
        turn_1 = [1, PAD, PAD, 2]
        turn_2 = [1, PAD, PAD, 2, 5, 6, 7]
        keyed_1 = vision_bank_key_ids(turn_1, splice)
        keyed_2 = vision_bank_key_ids(turn_2, make_splice([0xABCD], [2]))
        assert keyed_1 is not None and keyed_2 is not None
        assert keyed_2[: len(keyed_1)] == keyed_1

    def test_missing_identity_returns_none(self):
        prompt = [1, PAD, 2]
        splice = VisionSplice(image_pad_token_id=PAD, embeddings=mx.zeros((1, 8)))
        assert vision_bank_key_ids(prompt, splice) is None

    def test_pad_count_mismatch_returns_none(self):
        prompt = [1, PAD, 2]  # one pad in prompt
        splice = make_splice([0xABCD], [2])  # claims two
        assert vision_bank_key_ids(prompt, splice) is None

    def test_digest_padcount_length_mismatch_returns_none(self):
        prompt = [1, PAD, 2]
        splice = make_splice([0xA, 0xB], [1])
        assert vision_bank_key_ids(prompt, splice) is None


class TestFlashNextPositionSchemeKey:
    """The version salt of Flash-Next entries roped through the M-RoPE table.

    Builds 2.10.1 to 2.11.3 wrote those entries under the unsalted key, and
    their copy-round, repair and final-commit rows sit |delta| positions off.
    The key this build writes must never match one of them at or past an
    image, while every other key stays what it was.
    """

    PROMPT = (1, 2, PAD, PAD, PAD, PAD, 3, 4)
    DIGEST = 0x1234ABCD
    GRID = (1, 4, 4)

    def splice(self, *, table=True, delta=-2, grid=GRID, dense=None):
        return VisionSplice(
            image_pad_token_id=PAD,
            embeddings=mx.zeros((4, 8)),
            image_digests=(self.DIGEST,),
            pad_counts=(4,),
            image_grids=(grid,),
            mrope_table=mx.zeros((3, len(self.PROMPT)), dtype=mx.int32) if table else None,
            mrope_delta=delta,
            dense_mrope=dense,
        )

    def key_older_builds_wrote(self):
        """Written out by hand: digest and row index, nothing else."""
        from mtplx.vision.splice import _BANK_KEY_MASK, _BANK_KEY_MIX

        keyed = list(self.PROMPT)
        for row, pos in enumerate(i for i, t in enumerate(self.PROMPT) if t == PAD):
            keyed[pos] = _BANK_KEY_FLAG | (
                (self.DIGEST ^ (row * _BANK_KEY_MIX)) & _BANK_KEY_MASK
            )
        return keyed

    def test_salt_is_the_hashed_scheme_tag(self):
        import hashlib

        from mtplx.vision.mrope import BANK_KEY_SCHEME
        from mtplx.vision.splice import (
            _BANK_KEY_MASK,
            _DENSE_MROPE_KEY_SALT,
            _QWEN4_MROPE_KEY_SALT,
        )

        # A new scheme is a new tag and a new salt; keys written under this
        # one stay readable across processes and on disk.
        assert BANK_KEY_SCHEME == "qwen4_mrope_v2"
        digest = hashlib.blake2b(BANK_KEY_SCHEME.encode(), digest_size=8).digest()
        assert _QWEN4_MROPE_KEY_SALT == int.from_bytes(digest, "big") & _BANK_KEY_MASK
        assert _QWEN4_MROPE_KEY_SALT not in (0, _DENSE_MROPE_KEY_SALT)

    def test_roped_request_never_matches_the_key_older_builds_wrote(self):
        old = self.key_older_builds_wrote()
        new = vision_bank_key_ids(self.PROMPT, self.splice())
        pads = [i for i, token in enumerate(self.PROMPT) if token == PAD]
        # A prefix match can still reach the image through the shared text...
        assert all(new[i] == old[i] for i in range(len(self.PROMPT)) if i not in pads)
        # ...and never goes past its first row.
        assert all(new[i] != old[i] for i in pads)
        assert all(new[i] & _BANK_KEY_FLAG for i in pads)
        assert len({new[i] for i in pads}) == len(pads)
        # Stable: the same request keys the same way, turn after turn.
        assert vision_bank_key_ids(self.PROMPT, self.splice()) == new
        longer = [*self.PROMPT, 5, 6]
        table = mx.zeros((3, len(longer)), dtype=mx.int32)
        follow_up = self.splice()
        follow_up.mrope_table = table
        assert vision_bank_key_ids(longer, follow_up)[: len(new)] == new

    def test_sequentially_roped_image_request_keeps_the_key_it_always_had(self):
        # No table and no delta (a family without the contract, or a table
        # that could not be built): every forward ropes at the raw index, on
        # older builds too, so those entries are sound and stay reachable.
        sequential = vision_bank_key_ids(self.PROMPT, self.splice(table=False, delta=0))
        assert sequential == self.key_older_builds_wrote()

    def test_key_is_salted_exactly_when_the_attention_scope_is_armed(self):
        import contextlib

        from mtplx import generation
        from mtplx.attention_context import vision_rope_state

        old = self.key_older_builds_wrote()
        for table, delta in ((True, -2), (True, 0), (False, -2), (False, 0)):
            splice = self.splice(table=table, delta=delta)
            scope = generation._vision_rope_scope_for(splice)
            armed = not isinstance(scope, contextlib.nullcontext)
            with scope:
                assert (vision_rope_state() is not None) == armed
            salted = vision_bank_key_ids(self.PROMPT, splice) != old
            assert salted == armed == (table or delta != 0)

    def test_grid_and_scheme_are_part_of_the_key(self):
        from types import SimpleNamespace

        pads = [i for i, token in enumerate(self.PROMPT) if token == PAD]
        roped = vision_bank_key_ids(self.PROMPT, self.splice())
        other_grid = vision_bank_key_ids(self.PROMPT, self.splice(grid=(1, 2, 8)))
        assert all(roped[i] != other_grid[i] for i in pads)
        # The dense path keeps its own scheme: same pixels, other keys.
        dense = vision_bank_key_ids(
            self.PROMPT,
            self.splice(table=False, delta=0, dense=SimpleNamespace()),
        )
        assert all(dense[i] != roped[i] for i in pads)
        assert all(dense[i] != self.key_older_builds_wrote()[i] for i in pads)


class TestImageContentDigest:
    def test_digest_stable_and_content_sensitive(self):
        from mtplx.server.openai import _image_content_digest

        a = _image_content_digest(b"pixels-a")
        assert a == _image_content_digest(b"pixels-a")
        assert a != _image_content_digest(b"pixels-b")
        assert 0 <= a < (1 << 64)


class TestVisionEmbedCache:
    def test_rows_cached_by_digest_and_evicted_by_row_budget(self, monkeypatch):
        import mtplx.server.openai as srv

        calls = []

        import mtplx.vision as vision_pkg
        import mtplx.vision.processing as processing_pkg

        monkeypatch.setattr(
            vision_pkg, "load_vision_tower",
            lambda path: (lambda pv, grids: (mx.zeros((4, 8)), {})),
        )
        monkeypatch.setattr(processing_pkg, "decode_image", lambda raw: raw)
        monkeypatch.setattr(
            processing_pkg, "image_pad_token_count", lambda grid: 4
        )

        def counting_preprocess(imgs, cfg):
            calls.append(1)
            return mx.zeros((4, 3)), [(1, 4, 4)]

        monkeypatch.setattr(processing_pkg, "preprocess_images", counting_preprocess)
        srv._VISION_EMBED_CACHE.clear()

        rows_1, count_1, grid_1 = srv._vision_rows_for_image(None, "/m", {}, b"img-a", 111)
        rows_2, count_2, grid_2 = srv._vision_rows_for_image(None, "/m", {}, b"img-a", 111)
        assert count_1 == count_2 == 4
        assert grid_1 == grid_2 == (1, 4, 4), (
            "the digest cache must carry the (t, h, w) grid for M-RoPE"
        )
        assert len(calls) == 1, "second identical image must hit the digest cache"
        srv._vision_rows_for_image(None, "/m", {}, b"img-b", 222)
        assert len(calls) == 2

        monkeypatch.setattr(srv, "_VISION_EMBED_CACHE_MAX_ROWS", 4)
        srv._vision_rows_for_image(None, "/m", {}, b"img-c", 333)
        assert (str("/m"), 111) not in srv._VISION_EMBED_CACHE, (
            "row budget must evict the least recently used entry"
        )
        srv._VISION_EMBED_CACHE.clear()

    def test_kill_switch_disables_cache(self, monkeypatch):
        import mtplx.server.openai as srv

        monkeypatch.setenv("MTPLX_VISION_EMBED_CACHE", "0")
        assert not srv._vision_embed_cache_enabled()
        monkeypatch.setenv("MTPLX_VISION_EMBED_CACHE", "1")
        assert srv._vision_embed_cache_enabled()


class TestVisionSessionCacheFlag:
    def test_default_on_with_kill_switch(self, monkeypatch):
        import mtplx.server.openai as srv

        monkeypatch.delenv("MTPLX_VISION_SESSION_CACHE", raising=False)
        assert srv._vision_session_cache_enabled()
        monkeypatch.setenv("MTPLX_VISION_SESSION_CACHE", "0")
        assert not srv._vision_session_cache_enabled()


class TestRestoreGuard:
    def test_bank_without_identity_still_raises(self):
        from mtplx.generation import restore_or_prefill_prompt_state

        splice = VisionSplice(image_pad_token_id=PAD, embeddings=mx.zeros((1, 8)))

        class FakeBank:
            pass

        with pytest.raises(ValueError, match="content-keyed"):
            restore_or_prefill_prompt_state(
                None,  # runtime unused before the guard fires
                [1, PAD, 2],
                session_bank=FakeBank(),
                vision_splice=splice,
            )
