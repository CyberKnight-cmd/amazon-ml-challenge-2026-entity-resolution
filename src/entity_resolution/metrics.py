"""Emulation of the leaderboard metric: macro-averaged F0.5 over S1 entities.

Per entity: truth empty -> 1 if prediction empty else 0 (singleton rule);
truth non-empty -> P = |pred & truth| / |pred|, R = |pred & truth| / |truth|, F0.5 = 1.25*P*R / (0.25*P + R),
0 if the prediction is empty or nothing is correct. The organizers do not publish the per-entity definition;
this is the natural reading of the video and must be confirmed against the first leaderboard score.
"""

from __future__ import annotations

import polars as pl


def macro_f05(s1_ids: pl.DataFrame, truth: pl.DataFrame, pred: pl.DataFrame) -> dict:
    """s1_ids: column s1 (all entities scored). truth/pred: long tables with columns (s1, other)."""
    n_true = truth.group_by("s1").agg(pl.len().alias("n_true"))
    n_pred = pred.group_by("s1").agg(pl.len().alias("n_pred"))
    tp = (
        pred.join(truth.with_columns(pl.lit(1).alias("hit")), on=["s1", "other"], how="inner")
        .group_by("s1")
        .agg(pl.len().alias("tp"))
    )
    df = (
        s1_ids.select("s1")
        .unique()
        .join(n_true, on="s1", how="left")
        .join(n_pred, on="s1", how="left")
        .join(tp, on="s1", how="left")
        .with_columns(pl.col(["n_true", "n_pred", "tp"]).fill_null(0))
        .with_columns(
            (pl.col("tp") / pl.col("n_pred").clip(lower_bound=1)).alias("p"),
            (pl.col("tp") / pl.col("n_true").clip(lower_bound=1)).alias("r"),
        )
        .with_columns(
            pl.when(pl.col("n_true") == 0)
            .then(pl.when(pl.col("n_pred") == 0).then(1.0).otherwise(0.0))
            .when((pl.col("n_pred") == 0) | (pl.col("tp") == 0))
            .then(0.0)
            .otherwise(1.25 * pl.col("p") * pl.col("r") / (0.25 * pl.col("p") + pl.col("r")))
            .alias("f")
        )
    )
    single = df.filter(pl.col("n_true") == 0)
    multi = df.filter(pl.col("n_true") > 0)
    return {
        "f05": df["f"].mean(),
        "entities": df.height,
        "singleton_f": single["f"].mean() if single.height else None,
        "matched_f": multi["f"].mean() if multi.height else None,
        # over entities where we predicted something (an empty prediction is a recall failure, not a precision one)
        "matched_precision": multi.filter(pl.col("n_pred") > 0)["p"].mean() if multi.filter(pl.col("n_pred") > 0).height else None,
        "predicted_share": multi.filter(pl.col("n_pred") > 0).height / multi.height if multi.height else None,
        "matched_recall": multi["r"].mean() if multi.height else None,
    }
