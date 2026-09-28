"""Entity-context second stage: re-score each query's best assignment using what *other* queries say about the entity.

Motivation (measured on the full train world, see docs/07 and docs/11): most remaining loss is *missed records
inside entities we already found* (336k entities lose only recall). A weak record (p1 = 0.4) attached to an entity
that already owns several strong records is far more likely to be genuine than the same weak record attached to an
entity nobody else points at. The pair matcher cannot see that; this stage can.

Features for a query q whose best entity is e (all computed from the scored table, no labels needed):
    p1, p2, gap                       the query's own evidence and its competition
    n_strong / n_strong_s2 / _s3      how many OTHER queries also point at e with p1 >= 0.9 (total and per source)
    n_mid                             ... with p1 >= 0.5
    max_other                         the strongest other query pointing at e (0 if none)
    sum_other                         total p1 of other queries pointing at e
    leak                              probability other queries put on e as their *runner-up*
    same_src_strong                   strong records of the same source as q (vendors repeat their own noise)
    src                               0 = S2, 1 = S3
The output r is P(q really belongs to e), a calibrated probability that a plain threshold can use.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

STRONG, MID = 0.9, 0.5
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300, feature_fraction=0.9,
              verbose=-1, num_threads=8)
FEATURES = ["p1", "p2", "gap", "n_strong", "n_strong_s2", "n_strong_s3", "n_mid", "max_other", "sum_other", "leak",
            "same_src_strong", "src"]


def add_context(t2: pl.DataFrame) -> pl.DataFrame:
    """t2 = calibration.top2_with_ids output. Adds the entity-context feature columns (see module docstring)."""
    t = t2.with_columns(
        (pl.col("q_id").str.slice(1, 1) == "3").cast(pl.Int8).alias("src"),
        (pl.col("p1") - pl.col("p2")).alias("gap"),
        (pl.col("p1") >= STRONG).cast(pl.Int32).alias("_st"),
        (pl.col("p1") >= MID).cast(pl.Int32).alias("_mid"),
    )
    t = t.with_columns(
        ((pl.col("_st") == 1) & (pl.col("src") == 0)).cast(pl.Int32).alias("_st2"),
        ((pl.col("_st") == 1) & (pl.col("src") == 1)).cast(pl.Int32).alias("_st3"),
    )
    g = t.group_by("s1_id").agg(
        pl.col("_st").sum().alias("g_st"), pl.col("_st2").sum().alias("g_st2"), pl.col("_st3").sum().alias("g_st3"),
        pl.col("_mid").sum().alias("g_mid"), pl.col("p1").sum().alias("g_sum"),
        pl.col("p1").top_k(2).alias("_top2"),
    )
    g = g.with_columns(
        pl.col("_top2").list.get(0, null_on_oob=True).fill_null(0.0).alias("g_max1"),
        pl.col("_top2").list.get(1, null_on_oob=True).fill_null(0.0).alias("g_max2"),
    ).drop("_top2")
    leak = (
        t.select(pl.col("s1_id2").alias("s1_id"), pl.col("p2").alias("_l")).drop_nulls("s1_id")
        .group_by("s1_id").agg(pl.col("_l").sum().alias("g_leak"))
    )
    t = t.join(g, on="s1_id", how="left").join(leak, on="s1_id", how="left")
    is_self_max = pl.col("p1") >= pl.col("g_max1")
    return t.with_columns(
        (pl.col("g_st") - pl.col("_st")).alias("n_strong"),
        (pl.col("g_st2") - pl.col("_st2")).alias("n_strong_s2"),
        (pl.col("g_st3") - pl.col("_st3")).alias("n_strong_s3"),
        (pl.col("g_mid") - pl.col("_mid")).alias("n_mid"),
        pl.when(is_self_max).then(pl.col("g_max2")).otherwise(pl.col("g_max1")).alias("max_other"),
        (pl.col("g_sum") - pl.col("p1")).alias("sum_other"),
        pl.col("g_leak").fill_null(0.0).alias("leak"),
        pl.when(pl.col("src") == 0).then(pl.col("g_st2") - pl.col("_st2")).otherwise(pl.col("g_st3") - pl.col("_st3")).alias("same_src_strong"),
    ).drop([c for c in t.columns if c.startswith("_")] + ["g_st", "g_st2", "g_st3", "g_mid", "g_sum", "g_max1", "g_max2", "g_leak"])


def _matrix(t: pl.DataFrame) -> np.ndarray:
    """Model input matrix in the fixed FEATURES order, float32."""
    return t.select(pl.col(FEATURES).cast(pl.Float32)).to_numpy()


def fit_context_model(t: pl.DataFrame, truth_of_query: pl.DataFrame, rounds: int = 300) -> lgb.Booster:
    """t: add_context output (rows to train on); truth_of_query: (q_id, true_s1). Label = best entity is the owner."""
    d = t.join(truth_of_query, on="q_id", how="left")
    y = (d["s1_id"] == d["true_s1"]).fill_null(False).cast(pl.Int8).to_numpy()
    return lgb.train(PARAMS, lgb.Dataset(_matrix(d), y), rounds)


def apply_context_model(t: pl.DataFrame, model: lgb.Booster) -> pl.DataFrame:
    """Adds column r = P(query belongs to its best entity | own evidence and entity context)."""
    return t.with_columns(pl.Series("r", model.predict(_matrix(t)), dtype=pl.Float64))
