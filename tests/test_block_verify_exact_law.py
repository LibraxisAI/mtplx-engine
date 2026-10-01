"""Independent exactness oracle for the two accept laws.

Neither the shipped Leviathan-Chen loop nor block verification is trusted by
comparing it with a mirror of itself here.  Every draft path, every rejection
depth and every correction token is ENUMERATED on a tiny vocabulary and the
emitted joint law of the first ``D`` committed tokens is compared with the
target's joint law.  Tokens after a rejection (next-round primaries / the
bonus) are drawn from the target, exactly as the runtime does.

Exact fixtures from the 2026-09-07/08 audits (three independent
investigations found the pre-fix block law off by 1/64 to 4e-2 total
variation at depth 3) are pinned alongside random tables.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest

from mtplx.qwen4_block_verify import BlockVerifier, build_verifier, prepared_pair
from mtplx.sampling import (
    SparseDistribution,
    acceptance_probability,
    residual_distribution,
)


def _sparse(p: np.ndarray) -> SparseDistribution:
    ids = np.flatnonzero(p > 0)
    return SparseDistribution(ids, p[ids], p.shape[0])


def _dense(d: SparseDistribution, vocab: int) -> np.ndarray:
    out = np.zeros(vocab)
    out[d.token_ids] = d.probs
    return out


def _tables(rng, vocab, depth, mode):
    """Conditional target/draft rows p[d][prefix], q[d][prefix]."""
    P, Q = [], []
    for d in range(depth):
        Pd, Qd = {}, {}
        for prefix in itertools.product(range(vocab), repeat=d):
            z = rng.normal(size=vocab) * 1.5
            p = np.exp(z - z.max())
            p /= p.sum()
            if mode == "correlated":
                z = np.log(p) + rng.normal(size=vocab) * 0.7
            elif mode == "sharpdraft":
                z = np.log(p) * 3.0 + rng.normal(size=vocab) * 0.3
            else:
                z = rng.normal(size=vocab) * 1.5
            q = np.exp(z - z.max())
            q /= q.sum()
            if mode != "dense":
                # truncate like top-k=3 on both rows to mimic runtime supports
                for arr in (p, q):
                    order = np.argsort(-arr)
                    arr[order[3:]] = 0.0
                    arr /= arr.sum()
            Pd[prefix], Qd[prefix] = p, q
        P.append(Pd)
        Q.append(Qd)
    return P, Q


def _target_joint(P, vocab, depth):
    J = np.zeros((vocab,) * depth)
    for path in itertools.product(range(vocab), repeat=depth):
        pr = 1.0
        for d in range(depth):
            pr *= P[d][path[:d]][path[d]]
        J[path] = pr
    return J


def _fill_tail(J, weight, committed, P, vocab, depth):
    k = len(committed)
    if k == depth:
        J[tuple(committed)] += weight
        return
    for y in range(vocab):
        w = weight * P[k][tuple(committed)][y]
        if w > 0:
            _fill_tail(J, w, committed + [y], P, vocab, depth)


def _emitted_joint(P, Q, vocab, depth, law):
    """Enumerate the emitted joint law under ``law`` in {"shipped", "block"}."""
    J = np.zeros((vocab,) * depth)
    for path in itertools.product(range(vocab), repeat=depth):
        qpath = 1.0
        for d in range(depth):
            qpath *= Q[d][path[:d]][path[d]]
        if qpath == 0:
            continue
        if law == "block":
            bv = BlockVerifier(
                draft_tokens=list(path),
                draft_rows=[prepared_pair(_sparse(Q[d][path[:d]])) for d in range(depth)],
                target_rows=[prepared_pair(_sparse(P[d][path[:d]])) for d in range(depth)],
                vocab_size=vocab,
            )
        reach = qpath
        committed: list[int] = []
        for d in range(depth):
            p = _sparse(P[d][path[:d]])
            q = _sparse(Q[d][path[:d]])
            if law == "block":
                a = float(bv.accept_probability[d])
                residual = _dense(bv.scaled_residual(d), vocab)
            else:
                a = acceptance_probability(p, q, path[d])
                residual = _dense(residual_distribution(p, q), vocab) if a < 1.0 else None
            if a < 1.0:
                for y in range(vocab):
                    if residual[y] > 0:
                        _fill_tail(J, reach * (1 - a) * residual[y], committed + [y], P, vocab, depth)
            reach *= a
            committed = committed + [path[d]]
            if reach == 0:
                break
        else:
            _fill_tail(J, reach, committed, P, vocab, depth)
    return J


def _tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["dense", "correlated", "sharpdraft", "independent"])
def test_both_laws_emit_the_target_joint_law(depth, mode):
    rng = np.random.default_rng(1000 * depth + len(mode))
    vocab = 4 if depth == 4 else 5
    for _trial in range(4):
        P, Q = _tables(rng, vocab, depth, mode)
        target = _target_joint(P, vocab, depth)
        shipped = _emitted_joint(P, Q, vocab, depth, "shipped")
        block = _emitted_joint(P, Q, vocab, depth, "block")
        assert abs(shipped.sum() - 1.0) < 1e-12
        assert abs(block.sum() - 1.0) < 1e-12
        assert _tv(shipped, target) < 1e-12
        assert _tv(block, target) < 1e-12


def test_the_block_law_never_clips_a_coin():
    rng = np.random.default_rng(7)
    for _trial in range(200):
        vocab, depth = 3, 3
        P, Q = _tables(rng, vocab, depth, "dense")
        for path in itertools.product(range(vocab), repeat=depth):
            bv = BlockVerifier(
                draft_tokens=list(path),
                draft_rows=[prepared_pair(_sparse(Q[d][path[:d]])) for d in range(depth)],
                target_rows=[prepared_pair(_sparse(P[d][path[:d]])) for d in range(depth)],
                vocab_size=vocab,
            )
            assert sum(bv.clipped) == 0
            assert all(0.0 <= c <= 1.0 for c in bv.accept_probability)


def test_the_audit_counterexample_is_exact_now():
    """Codex 2026-09-08: independent rows p=[(1/4,3/4),(1/2,1/2),(3/4,1/4)],
    q=[(3/4,1/4),(3/4,1/4),(1/4,3/4)] gave P(000)=5/64 (target 6/64) and
    P(001)=3/64 (target 2/64) under the pre-fix law."""

    p = [[Fraction(1, 4), Fraction(3, 4)], [Fraction(1, 2), Fraction(1, 2)], [Fraction(3, 4), Fraction(1, 4)]]
    q = [[Fraction(3, 4), Fraction(1, 4)], [Fraction(3, 4), Fraction(1, 4)], [Fraction(1, 4), Fraction(3, 4)]]
    sequences = list(itertools.product((0, 1), repeat=3))
    expected = {seq: p[0][seq[0]] * p[1][seq[1]] * p[2][seq[2]] for seq in sequences}
    actual = dict.fromkeys(sequences, 0.0)
    for drafted in sequences:
        bv = build_verifier(
            draft_tokens=drafted,
            draft_probs=[[float(v) for v in row] for row in q],
            target_list=[[float(v) for v in row] for row in p],
        )
        remaining = float(q[0][drafted[0]] * q[1][drafted[1]] * q[2][drafted[2]])
        for pos in range(3):
            coin = bv.accept_probability[pos]
            rejected = remaining * (1 - coin)
            residual = bv.scaled_residual(pos).to_dense()
            for output in sequences:
                if output[:pos] != drafted[:pos]:
                    continue
                probability = rejected * residual[output[pos]]
                for future in range(pos + 1, 3):
                    probability *= float(p[future][output[future]])
                actual[output] += probability
            remaining *= coin
        actual[drafted] += remaining
    assert abs(sum(actual.values()) - 1.0) < 1e-12
    for seq in sequences:
        assert abs(actual[seq] - float(expected[seq])) < 1e-12, (seq, actual[seq], expected[seq])


def test_the_gpt_pro_counterexample_is_exact_now():
    """ChatGPT Pro 2026-09-08: P1=(.25,.75) P2=(.40,.60) P3=(.60,.40), Q=(.5,.5)
    everywhere gave 0.01 total variation under the pre-fix law."""

    vocab, depth = 2, 3
    rows_p = [np.array([0.25, 0.75]), np.array([0.40, 0.60]), np.array([0.60, 0.40])]
    P = [{prefix: rows_p[d] for prefix in itertools.product(range(vocab), repeat=d)} for d in range(depth)]
    Q = [{prefix: np.array([0.5, 0.5]) for prefix in itertools.product(range(vocab), repeat=d)} for d in range(depth)]
    target = _target_joint(P, vocab, depth)
    block = _emitted_joint(P, Q, vocab, depth, "block")
    assert _tv(block, target) < 1e-12
