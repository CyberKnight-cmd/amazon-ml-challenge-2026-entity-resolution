import itertools
import math

import numpy as np
import polars as pl

from entity_resolution.expected_f05 import L, M, best_k, decode


def brute_expected(r_sorted, mu, k):
    """E[F0.5] of choosing the first k candidates, by enumerating every outcome (Poisson part truncated at L-1)."""
    r_sorted = list(r_sorted)
    total = 0.0
    pois = [math.exp(-mu) * mu**j / math.factorial(j) for j in range(L)]
    pois[-1] += 1 - sum(pois)
    for bits in itertools.product([0, 1], repeat=len(r_sorted)):
        pb = np.prod([r if b else 1 - r for r, b in zip(r_sorted, bits)])
        tp = sum(bits[:k])
        rest = sum(bits[k:])
        for u, pu in enumerate(pois):
            n = tp + rest + u
            f = (1.0 if n == 0 else 0.0) if k == 0 else 1.25 * tp / (0.25 * n + k)
            total += pb * pu * f
    return total


def test_dp_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(5):
        m = int(rng.integers(1, 6))
        r = np.sort(rng.uniform(0.05, 0.99, m))[::-1]
        mu = float(rng.uniform(0.01, 0.6))
        padded = np.zeros((1, M)); padded[0, :m] = r
        k_star, val = best_k(padded, np.array([mu]))
        brute = [brute_expected(r, mu, k) for k in range(m + 1)]
        assert abs(val[0] - max(brute)) < 1e-9
        assert int(k_star[0]) == int(np.argmax(brute))


def test_singleton_protection_and_confident_match():
    padded = np.zeros((3, M))
    padded[0, 0] = 0.98                     # a very confident single record
    padded[1, 0] = 0.30                     # a weak single record: better to predict nothing
    padded[2, :3] = [0.95, 0.9, 0.2]        # two confident + one doubtful
    k, _ = best_k(padded, np.full(3, 0.05))
    assert k.tolist() == [1, 0, 2]


def test_decode_returns_top_k_records_per_entity():
    assign = pl.DataFrame({"q_id": ["q1", "q2", "q3", "q4"], "s1_id": ["A", "A", "B", "B"], "r": [0.97, 0.95, 0.30, 0.25]})
    ru = pl.DataFrame({"s1_id": ["A", "B"], "mass": [0.0, 0.0]})
    acc = decode(assign, ru, mu0=0.05).sort("q_id")
    assert acc["q_id"].to_list() == ["q1", "q2"]  # B is too doubtful and is left empty
