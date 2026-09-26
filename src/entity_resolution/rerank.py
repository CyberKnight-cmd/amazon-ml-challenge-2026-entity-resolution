"""Second-stage re-ranker: re-score the plausible pairs (stage-1 p >= KEEP_P) with richer, costlier evidence.

Stage 1 (the pair matcher) scores ~28 candidates per query with cheap features. Only ~1 pair per query survives
its p >= 0.02 cut, so stage 2 can afford features that would be too slow for every candidate:

  * stage-1 probability and the query's competition (rank, strongest other candidate, number of candidates);
  * entity context from the stage-1 winners (how many other queries already point at the entity strongly,
    per source; excluding the pair itself) - the idea of context.py, available to every pair, not only the best;
  * the learned noisy-channel events of channel.py (typo-like vs. decoy-like differences);
  * how common the query's leftover name words are among S1 names (a real word substituted for a real word is a
    decoy signature; an unknown string is a typo or an invented trade name);
  * house-number relations (absolute difference, prefix/truncation);
  * the plain text similarities of features.py (recomputed; blocking statistics are summarised by stage-1 p).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import channel
from .features import _chunk_features

STRONG, MID = 0.9, 0.5
_BLOCK_COLS = ("n_keys", "score", "max_idf", "k_name", "k_addr", "k_bigram", "k_sq", "k_pair", "k_numpair", "k_addrbag")
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16, max_bin=255)


def query_context(scored: pl.DataFrame) -> pl.DataFrame:
    """Per pair: rank of p within its query, strongest *other* candidate of the query, candidates and mass per query."""
    s = scored.sort(["q_id", "p"], descending=[False, True]).with_columns(
        pl.int_range(pl.len()).over("q_id").alias("q_rank"),
        pl.len().over("q_id").alias("q_ncand"),
        pl.col("p").sum().over("q_id").alias("q_psum"),
        pl.col("p").first().over("q_id").alias("_p1"),
        pl.col("p").get(1, null_on_oob=True).fill_null(0.0).over("q_id").alias("_p2"),
    )
    return s.with_columns(
        pl.when(pl.col("q_rank") == 0).then(pl.col("_p2")).otherwise(pl.col("_p1")).alias("q_pother"),
    ).drop("_p1", "_p2")


def entity_context(s: pl.DataFrame) -> pl.DataFrame:
    """Entity aggregates over the queries whose stage-1 winner is the entity, minus the pair's own contribution."""
    src3 = pl.col("q_id").str.slice(1, 1) == "3"
    top = s.filter(pl.col("q_rank") == 0).with_columns(src3.alias("_s3"))
    g = top.group_by("s1_id").agg(
        (pl.col("p") >= STRONG).sum().alias("_st"),
        ((pl.col("p") >= STRONG) & ~pl.col("_s3")).sum().alias("_st2"),
        ((pl.col("p") >= STRONG) & pl.col("_s3")).sum().alias("_st3"),
        (pl.col("p") >= MID).sum().alias("_mid"),
        pl.col("p").sum().alias("_sum"),
        pl.len().alias("_n"),
        pl.col("p").top_k(2).alias("_top2"),
    ).with_columns(
        pl.col("_top2").list.get(0, null_on_oob=True).fill_null(0.0).alias("_m1"),
        pl.col("_top2").list.get(1, null_on_oob=True).fill_null(0.0).alias("_m2"),
    ).drop("_top2")
    allp = s.group_by("s1_id").agg(pl.col("p").sum().alias("_all"), pl.len().alias("_alln"))
    s = s.join(g, on="s1_id", how="left").join(allp, on="s1_id", how="left").with_columns(
        pl.col(c).fill_null(0) for c in ("_st", "_st2", "_st3", "_mid", "_sum", "_n", "_m1", "_m2")
    )
    me = (pl.col("q_rank") == 0).cast(pl.Int32)
    st = me * (pl.col("p") >= STRONG).cast(pl.Int32)
    s3 = src3.cast(pl.Int32)
    return s.with_columns(
        (pl.col("_st") - st).alias("e_strong"),
        (pl.col("_st2") - st * (1 - s3)).alias("e_strong_s2"),
        (pl.col("_st3") - st * s3).alias("e_strong_s3"),
        (pl.col("_mid") - me * (pl.col("p") >= MID).cast(pl.Int32)).alias("e_mid"),
        (pl.col("_sum") - me * pl.col("p")).alias("e_sum"),
        (pl.col("_n") - me).alias("e_n"),
        pl.when((me == 1) & (pl.col("p") >= pl.col("_m1"))).then(pl.col("_m2")).otherwise(pl.col("_m1")).alias("e_max"),
        (pl.col("_all") - pl.col("p")).alias("e_allp"),
        (pl.col("_alln") - 1).alias("e_alln"),
        pl.when(s3 == 1).then(pl.col("_st3") - st).otherwise(pl.col("_st2") - st).alias("e_same_src_strong"),
        s3.alias("src3"),
    ).drop("_st", "_st2", "_st3", "_mid", "_sum", "_n", "_m1", "_m2", "_all", "_alln")


def name_vocab(s1: pl.DataFrame) -> pl.DataFrame:
    """Document frequency of every name token among the country's S1 entities."""
    return (
        s1.select("id", pl.col("name_n").str.split(" ").alias("t")).explode("t").filter(pl.col("t") != "")
        .unique().group_by("t").agg(pl.len().alias("vdf"))
    )


def _vocab_feats(p: pl.DataFrame, vocab: pl.DataFrame) -> pl.DataFrame:
    """For the query's leftover name tokens: min / max S1 frequency and how many are unknown to S1 names."""
    tok = lambda c: pl.col(c).str.split(" ")  # noqa: E731
    x = p.select(pl.int_range(pl.len()).alias("_r"), tok("n2").list.set_difference(tok("n1")).alias("t")).explode("t")
    x = x.filter(pl.col("t").is_not_null() & (pl.col("t") != "")).join(vocab, on="t", how="left").with_columns(
        pl.col("vdf").fill_null(0)
    )
    g = x.group_by("_r").agg(
        (pl.col("vdf") + 1).log().min().alias("v_min"),
        (pl.col("vdf") + 1).log().max().alias("v_max"),
        (pl.col("vdf") == 0).sum().alias("v_unknown"),
        (pl.col("vdf") > 0).sum().alias("v_known"),
    )
    return pl.DataFrame({"_r": np.arange(p.height, dtype=np.int64)}).join(
        g.with_columns(pl.col("_r").cast(pl.Int64)), on="_r", how="left").drop("_r")


def _number_feats(p: pl.DataFrame) -> pl.DataFrame:
    """Relation of the first house-number-like token of each address."""
    h1 = pl.col("a1").str.extract(r"\b(\d+)\b", 1)
    h2 = pl.col("a2").str.extract(r"\b(\d+)\b", 1)
    x = p.select(h1.alias("h1"), h2.alias("h2"))
    v1, v2 = pl.col("h1").cast(pl.Float64, strict=False), pl.col("h2").cast(pl.Float64, strict=False)
    return x.select(
        (v1 - v2).abs().log1p().cast(pl.Float32).alias("h_absdiff"),
        ((v1 + 1).log() - (v2 + 1).log()).abs().cast(pl.Float32).alias("h_logratio"),
        (pl.col("h1").str.starts_with(pl.col("h2")) | pl.col("h2").str.starts_with(pl.col("h1"))).cast(pl.Float32).alias("h_prefix"),
        (pl.col("h1").str.ends_with(pl.col("h2")) | pl.col("h2").str.ends_with(pl.col("h1"))).cast(pl.Float32).alias("h_suffix"),
        (pl.col("h1").str.len_chars() - pl.col("h2").str.len_chars()).cast(pl.Float32).alias("h_lendiff"),
    )


def base_frame(scored: pl.DataFrame, s1s: pl.DataFrame, queries: pl.DataFrame) -> pl.DataFrame:
    """scored: q_id, s1_id, p (stage 1). s1s: features.prepare_s1(s1). queries: prep_queries output.
    Returns one row per pair with ids, stage-1/context columns and the raw texts n1,a1,n2,a2 (for later steps)."""
    s = entity_context(query_context(scored))
    qs = queries.select(
        pl.col("id").alias("q_id"), pl.col("name_n").alias("n2"), pl.col("addr_n").alias("a2"),
        pl.col("name_sq").alias("q2"), pl.col("is_domain").alias("d2"),
        pl.col("native_name").cast(pl.Float32).alias("q_native_name"),
        pl.col("native_addr").cast(pl.Float32).alias("q_native_addr"),
        (pl.col("id").str.slice(1, 1) == "3").cast(pl.Float32).alias("src"),
    )
    return s.join(s1s, on="s1_id").join(qs, on="q_id")


def text_features(b: pl.DataFrame, vocab: pl.DataFrame) -> pl.DataFrame:
    """Plain similarity features (features.py), vocabulary and house-number features for a base_frame chunk."""
    p = b.with_columns([pl.lit(0).alias(c) for c in _BLOCK_COLS])
    f = _chunk_features(p).drop(["q_id", "s1_id", *_BLOCK_COLS])
    return pl.concat([f, _vocab_feats(b, vocab), _number_feats(b)], how="horizontal")


CTX_COLS = ["p", "q_rank", "q_ncand", "q_psum", "q_pother", "e_strong", "e_strong_s2", "e_strong_s3", "e_mid", "e_sum",
            "e_n", "e_max", "e_allp", "e_alln", "e_same_src_strong", "src3"]
