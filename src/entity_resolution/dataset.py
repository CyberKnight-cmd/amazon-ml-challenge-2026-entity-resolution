"""Build labelled pair tables from train: full S1 index per country, sampled S2/S3 queries."""

from __future__ import annotations

import os

import polars as pl

from .aliases import add_native_flags, apply_aliases
from .blocking import BlockingConfig, generate_candidates
from .features import pair_features, prepare_s1
from .io import WORK, load_normalized, load_truth

N_FOLDS = 5


def alias_mode() -> str:
    """Only cross-script (native) aliases help: Latin-to-Latin rewrites measurably hurt (see docs/PLAN.md)."""
    return os.environ.get("ER_ALIAS_MODE", "native")


def load_alias_table() -> pl.DataFrame | None:
    """Read the fitted alias table from $ER_WORK, or None if it has not been fitted yet."""
    path = WORK / "aliases.parquet"
    return pl.read_parquet(path) if path.exists() else None


def prep_queries(q: pl.DataFrame, table: pl.DataFrame | None) -> pl.DataFrame:
    """Flag native-script text, then rewrite query tokens to the S1 spelling using the learned alias table."""
    q = add_native_flags(q)
    return apply_aliases(q, table, alias_mode()) if table is not None else q


def build_pair_table(country: str, n_queries: int, seed: int = 7, cfg: BlockingConfig = BlockingConfig()) -> pl.DataFrame:
    """Labelled candidate pairs for one country: sample `n_queries` S2/S3 records, block against the full S1 index,
        compute features, and label each pair (1 = true owner). Also writes a per-query meta table incl. fold and truth."""
    s1 = load_normalized("train", 1).filter(pl.col("country") == country)
    rec = pl.concat([load_normalized("train", 2), load_normalized("train", 3)]).filter(pl.col("country") == country)
    q = prep_queries(rec.sample(n_queries, seed=seed), load_alias_table())
    truth = load_truth("train").rename({"other": "q_id", "s1": "true_s1"})

    cand = generate_candidates(s1, q, cfg)
    feats = pair_features(cand, prepare_s1(s1), q)
    qt = q.select(pl.col("id").alias("q_id")).join(truth, on="q_id", how="left")
    # fold is a property of the *true entity* (distractors: of the query) so no entity spans train and validation
    qt = qt.with_columns(
        (pl.coalesce("true_s1", "q_id").hash(seed=13) % N_FOLDS).cast(pl.Int8).alias("fold"),
    )
    qt.with_columns(pl.lit(country).alias("country")).write_parquet(WORK / f"meta_{country}_{n_queries}_{seed}.parquet")
    out = feats.join(qt, on="q_id").with_columns(
        (pl.col("s1_id") == pl.col("true_s1")).cast(pl.Int8).alias("label"),
        pl.lit(country).alias("country"),
    )
    return out


def cached_pair_table(country: str, n_queries: int, seed: int = 7) -> pl.DataFrame:
    """`build_pair_table` with an on-disk cache keyed by country, sample size, seed and alias mode."""
    WORK.mkdir(parents=True, exist_ok=True)
    mode = alias_mode()
    path = WORK / f"pairs_{country}_{n_queries}_{seed}_{mode}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    WORK.mkdir(parents=True, exist_ok=True)
    df = build_pair_table(country, n_queries, seed)
    df.write_parquet(path, compression="zstd")
    return df
