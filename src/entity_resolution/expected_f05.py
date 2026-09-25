"""Expected-F0.5 decoding: choose, for every S1 entity, the set of records that maximizes its *expected* score.

Why: the leaderboard scores each S1 entity separately with F0.5, and a singleton is all-or-nothing. Deciding
record by record with one global threshold ignores that. Here we decide per entity, using calibrated probabilities.

Setup for one entity e
    * candidates i = 1..M : queries (S2/S3 records) whose best S1 candidate is e, with calibrated probability r_i
      that the query really belongs to e. We sort them by r_i (descending); the optimal answer is a prefix.
    * U : true matches of e that are NOT among the candidates (queries whose best guess is another entity,
      blocking misses, ...). Modelled as Poisson(mu_e), with mu_e = (probability mass other queries put on e as
      their runner-up) + a constant prior mu0.
    * Outcome: TP = number of chosen candidates that are true, N = TP + (true among the rest) + U.

Per-entity F0.5 for a chosen set S (size k):
        k >= 1 :  1.25 * TP / (0.25 * N + k)          (this is 1.25*P*R / (0.25*P + R) after simplification)
        k == 0 :  1 if N == 0 else 0                   (singleton rule)
We compute E[F0.5] exactly for every k with two Poisson-binomial distributions (chosen part, remaining part) and
keep the best k. Everything is vectorized over entities. Assumption: records are independent given their
probabilities (approximate but cheap; calibration is what matters most).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

M = 10   # most candidates considered per entity
L = 6    # support size of the Poisson part
R_LEN = M + L


def _weights(k: int) -> np.ndarray:
    """W[tp, rr] = F0.5 when tp of the k chosen are true and rr true matches lie outside the chosen set."""
    tp = np.arange(M + 1)[:, None].astype(np.float64)
    rr = np.arange(R_LEN)[None, :].astype(np.float64)
    if k == 0:
        return np.broadcast_to((rr == 0).astype(np.float64), (M + 1, R_LEN)).copy() * (tp == 0)
    return 1.25 * tp / (0.25 * (tp + rr) + k)


def _bernoulli_step(pmf: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Add one Bernoulli(r) variable to a batch of count distributions (Poisson-binomial recursion)."""
    out = pmf * (1.0 - r[:, None])
    out[:, 1:] += pmf[:, :-1] * r[:, None]
    return out


def _poisson_pmf(mu: np.ndarray, length: int) -> np.ndarray:
    """Poisson(mu) probability mass over 0..length-1 for each entity; the tail is folded into the last bin."""
    j = np.arange(length)[None, :]
    logf = np.array([math.lgamma(x + 1) for x in range(length)])[None, :]
    mu = np.maximum(mu, 1e-12)[:, None]
    pmf = np.exp(-mu + j * np.log(mu) - logf)
    pmf[:, -1] += np.clip(1.0 - pmf.sum(1), 0.0, None)  # fold the tail into the last bin
    return pmf


def best_k(r: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """r: (E, M) probabilities sorted descending, zero padded. mu: (E,). Returns (best k per entity, E[F0.5] at best k)."""
    E = r.shape[0]
    prefix = [np.zeros((E, M + 1))]
    prefix[0][:, 0] = 1.0
    for k in range(1, M + 1):
        prefix.append(_bernoulli_step(prefix[-1], r[:, k - 1]))
    suffix = np.zeros((E, R_LEN))
    suffix[:, :L] = _poisson_pmf(mu, L)
    scores = np.zeros((E, M + 1))
    for k in range(M, -1, -1):
        scores[:, k] = ((prefix[k] @ _weights(k)) * suffix).sum(1)
        if k >= 1:
            suffix = _bernoulli_step(suffix, r[:, k - 1])
    k_star = scores.argmax(1)  # ties resolve to the smaller k (more conservative)
    return k_star, scores[np.arange(E), k_star]


def decode(assign: pl.DataFrame, runner_up: pl.DataFrame, mu0: float, min_r: float = 0.02) -> pl.DataFrame:
    """assign: (q_id, s1_id, r)  one row per query = its best entity with calibrated probability.
    runner_up: (s1_id, mass)    expected number of true matches of s1_id sitting on other queries' second choice.
    Returns accepted (q_id, s1_id)."""
    a = (
        assign.filter(pl.col("r") >= min_r)
        .sort(["s1_id", "r"], descending=[False, True])
        .with_columns(pl.int_range(pl.len()).over("s1_id").alias("rank"))
        .filter(pl.col("rank") < M)
    )
    ent = a.select("s1_id").unique(maintain_order=True).with_row_index("e").join(runner_up, on="s1_id", how="left")
    ent = ent.with_columns(pl.col("mass").fill_null(0.0))
    a = a.join(ent.select("s1_id", "e"), on="s1_id")
    E = ent.height
    r = np.zeros((E, M))
    r[a["e"].to_numpy(), a["rank"].to_numpy()] = a["r"].to_numpy()
    k_star, _ = best_k(r, ent["mass"].to_numpy() + mu0)
    keep = ent.select("s1_id").with_columns(pl.Series("k", k_star))
    return a.join(keep, on="s1_id").filter(pl.col("rank") < pl.col("k")).select("q_id", "s1_id")
