"""Second-stage calibration: from the matcher's raw scores to "how likely is this assignment correct?".

The pair matcher scores each (query, entity) candidate on its own. Two things it cannot see:
  * competition - a query with p1=0.9 and p2=0.88 is far less certain than p1=0.9, p2=0.02;
  * selection - the best of ~28 candidates is optimistic compared with a random candidate.
So we fit two tiny monotone models on labelled train data:
  r1 = P(query's best candidate is its true entity | p1, p2)
  r2 = P(query's runner-up is its true entity      | p1, p2)
r1 feeds the expected-F0.5 decoder as the candidate probability; r2 (summed per entity) is the probability mass
that "leaks" to an entity through other queries' second choices.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=500, verbose=-1, num_threads=8)


def top2_with_ids(scored: pl.DataFrame) -> pl.DataFrame:
    """One row per query: best entity + p1, runner-up entity + p2 (null / 0 if there is none)."""
    g = scored.sort(["q_id", "p"], descending=[False, True]).group_by("q_id", maintain_order=True)
    return g.agg(
        pl.col("s1_id").first().alias("s1_id"),
        pl.col("p").first().alias("p1"),
        pl.col("s1_id").get(1, null_on_oob=True).alias("s1_id2"),
        pl.col("p").get(1, null_on_oob=True).fill_null(0.0).alias("p2"),
    )


def _x(t2: pl.DataFrame) -> np.ndarray:
    """Model input matrix (p1, p2) as float32."""
    return t2.select(pl.col("p1").cast(pl.Float32), pl.col("p2").cast(pl.Float32)).to_numpy()


def fit_calibrators(t2: pl.DataFrame, truth_of_query: pl.DataFrame) -> tuple[lgb.Booster, lgb.Booster]:
    """t2 from top2_with_ids; truth_of_query: (q_id, true_s1). Trained only on the rows given (caller splits)."""
    d = t2.join(truth_of_query, on="q_id", how="left")
    y1 = (d["s1_id"] == d["true_s1"]).fill_null(False).cast(pl.Int8).to_numpy()
    y2 = (d["s1_id2"] == d["true_s1"]).fill_null(False).cast(pl.Int8).to_numpy()
    x = _x(d)
    m1 = lgb.train({**PARAMS, "monotone_constraints": [1, -1]}, lgb.Dataset(x, y1), 200)
    has2 = d["s1_id2"].is_not_null().to_numpy()
    m2 = lgb.train({**PARAMS, "monotone_constraints": [-1, 1]}, lgb.Dataset(x[has2], y2[has2]), 200)
    return m1, m2


def apply_calibrators(t2: pl.DataFrame, m1: lgb.Booster, m2: lgb.Booster) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (assign: q_id, s1_id, r) and (runner_up: s1_id, mass) ready for expected_f05.decode."""
    x = _x(t2)
    r1 = m1.predict(x)
    r2 = np.where(t2["s1_id2"].is_not_null().to_numpy(), m2.predict(x), 0.0)
    assign = t2.select("q_id", "s1_id").with_columns(pl.Series("r", r1, dtype=pl.Float64))
    ru = (
        t2.select("s1_id2").with_columns(pl.Series("r2", r2))
        .drop_nulls("s1_id2").group_by("s1_id2").agg(pl.col("r2").sum().alias("mass"))
        .rename({"s1_id2": "s1_id"})
    )
    return assign, ru
