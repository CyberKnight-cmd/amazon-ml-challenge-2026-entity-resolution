"""Turn pair probabilities into assignments: each query picks its best S1 candidate or nothing."""

from __future__ import annotations

import polars as pl


def top2(scored: pl.DataFrame) -> pl.DataFrame:
    """One row per query: best candidate, its probability p1, and the runner-up probability p2 (0 if none)."""
    ranked = scored.sort(["q_id", "p"], descending=[False, True])
    g = ranked.group_by("q_id", maintain_order=True)
    return g.agg(
        pl.col("s1_id").first().alias("s1_id"),
        pl.col("p").first().alias("p1"),
        pl.col("p").get(1, null_on_oob=True).fill_null(0.0).alias("p2"),
    )


def accept(t2: pl.DataFrame, t: float, gap: float) -> pl.DataFrame:
    """Threshold decoder: keep queries with p1 >= t and a margin p1 - p2 >= gap. Returns (q_id, s1_id)."""
    return t2.filter((pl.col("p1") >= t) & ((pl.col("p1") - pl.col("p2")) >= gap)).select("q_id", "s1_id")
