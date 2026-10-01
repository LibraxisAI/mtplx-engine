"""The copy-lane probe reads prompt + generated tokens without concatenating them."""

from __future__ import annotations

import random

import pytest

from mtplx.context_copy import JoinedTokens, NgramIndex


def test_indexing_and_slicing_match_the_concatenated_list():
    rng = random.Random(7)
    head = [rng.randrange(1000) for _ in range(257)]
    tail = [rng.randrange(1000) for _ in range(19)]
    joined, plain = JoinedTokens(head, tail), head + tail
    assert len(joined) == len(plain)
    for index in (*range(-len(plain), len(plain)),):
        assert joined[index] == plain[index]
    for piece in (slice(-6, None), slice(250, 262), slice(None, 5), slice(0, None, 37), slice(270, 400)):
        assert joined[piece] == plain[piece]
    with pytest.raises(IndexError):
        joined[len(plain)]
    with pytest.raises(IndexError):
        joined[-len(plain) - 1]


def test_the_view_tracks_tokens_appended_after_it_was_built():
    head, tail = [1, 2, 3], [4]
    joined = JoinedTokens(head, tail)
    tail.append(5)
    assert len(joined) == 5 and joined[-1] == 5


@pytest.mark.parametrize("seed", range(6))
def test_the_index_answers_the_same_for_the_view_and_the_list(seed):
    rng = random.Random(seed)
    # A prompt with a repeated passage, so the 6-gram index has real matches.
    passage = [rng.randrange(50) for _ in range(40)]
    prompt = [rng.randrange(50) for _ in range(300)] + passage + [rng.randrange(50) for _ in range(200)] + passage
    index = NgramIndex(6, 10)
    index.sync(prompt)
    for cut in range(8, 40):
        generated = passage[:cut]
        plain = index.find(prompt + generated, max_pos=len(prompt))
        lazy = index.find(JoinedTokens(prompt, generated), max_pos=len(prompt))
        assert lazy == plain
