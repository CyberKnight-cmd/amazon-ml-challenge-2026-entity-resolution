"""Pair features for the matcher. String similarities run in rapidfuzz's multithreaded C++ (`cpdist`);
token/number overlaps run as polars expressions. All features are language-agnostic (no country input)."""

from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

CHUNK = 1_500_000

_NAME_SCORERS = {
    "n_ratio": fuzz.ratio,
    "n_tsort": fuzz.token_sort_ratio,
    "n_tset": fuzz.token_set_ratio,
    "n_jw": JaroWinkler.normalized_similarity,
}
_SQ_SCORERS = {"sq_ratio": fuzz.ratio, "sq_partial": fuzz.partial_ratio}
_ADDR_SCORERS = {
    "a_ratio": fuzz.ratio,
    "a_tsort": fuzz.token_sort_ratio,
    "a_tset": fuzz.token_set_ratio,
    "a_partial": fuzz.partial_token_set_ratio,
}


def _cp(a: list[str], b: list[str], scorer) -> np.ndarray:
    """Pairwise scores between two parallel string lists, computed multithreaded in rapidfuzz's C++ core."""
    return process.cpdist(a, b, scorer=scorer, dtype=np.float32, workers=-1)


def s1_ambiguity(s1: pl.DataFrame) -> pl.DataFrame:
    """How many S1 entities of the same country share this exact name / address (ambiguity of the reference)."""
    nm = s1.group_by(["country", "name_n"]).agg(pl.len().alias("s1_name_dup"))
    ad = s1.group_by(["country", "addr_n"]).agg(pl.len().alias("s1_addr_dup"))
    return (
        s1.join(nm, on=["country", "name_n"]).join(ad, on=["country", "addr_n"]).select("id", "s1_name_dup", "s1_addr_dup")
    )


def _chunk_features(p: pl.DataFrame) -> pl.DataFrame:
    """p has, per pair: n1,a1,q1,d1 (S1 side) and n2,a2,q2,d2 (query side) plus blocking stats."""
    n1, n2 = p["n1"].to_list(), p["n2"].to_list()
    a1, a2 = p["a1"].to_list(), p["a2"].to_list()
    q1, q2 = p["q1"].to_list(), p["q2"].to_list()
    cols: dict[str, np.ndarray] = {}
    for k, s in _NAME_SCORERS.items():
        cols[k] = _cp(n1, n2, s)
    for k, s in _SQ_SCORERS.items():
        cols[k] = _cp(q1, q2, s)
    for k, s in _ADDR_SCORERS.items():
        cols[k] = _cp(a1, a2, s)
    cols["a_jw"] = _cp(a1, a2, JaroWinkler.normalized_similarity)

    nums = lambda c: pl.col(c).str.extract_all(r"\d+")  # noqa: E731
    tok = lambda c: pl.col(c).str.split(" ")  # noqa: E731
    pure = lambda c: pl.col(c).str.extract_all(r"\b\d+\b")  # noqa: E731  (house-number-like tokens)
    # "leftover" analysis: what remains after removing shared tokens. A typo leaves two similar strings,
    # a substituted word / changed house number leaves two dissimilar ones.
    rest = p.select(
        tok("n1").list.set_difference(tok("n2")).list.sort().list.join(" ").alias("nr1"),
        tok("n2").list.set_difference(tok("n1")).list.sort().list.join(" ").alias("nr2"),
        tok("a1").list.set_difference(tok("a2")).list.sort().list.join(" ").alias("ar1"),
        tok("a2").list.set_difference(tok("a1")).list.sort().list.join(" ").alias("ar2"),
        pure("a1").list.set_difference(pure("a2")).list.sort().list.join(" ").alias("mr1"),
        pure("a2").list.set_difference(pure("a1")).list.sort().list.join(" ").alias("mr2"),
        tok("n1").list.set_intersection(tok("n2")).list.len().alias("n_shared_tok"),
        tok("a1").list.set_intersection(tok("a2")).list.len().alias("a_shared_tok"),
        tok("n1").list.len().alias("n1_ntok"), tok("n2").list.len().alias("n2_ntok"),
        tok("a1").list.len().alias("a1_ntok"), tok("a2").list.len().alias("a2_ntok"),
    )
    for name, (x, y) in {"n_rest": ("nr1", "nr2"), "a_rest": ("ar1", "ar2"), "m_rest": ("mr1", "mr2")}.items():
        cols[name + "_ratio"] = _cp(rest[x].to_list(), rest[y].to_list(), fuzz.ratio)
        cols[name + "_len1"] = rest[x].str.len_chars().to_numpy().astype(np.float32)
        cols[name + "_len2"] = rest[y].str.len_chars().to_numpy().astype(np.float32)
    cols["m_rest_dist"] = _cp(rest["mr1"].to_list(), rest["mr2"].to_list(), Levenshtein.distance)
    for c in ("n_shared_tok", "a_shared_tok", "n1_ntok", "n2_ntok", "a1_ntok", "a2_ntok"):
        cols[c] = rest[c].to_numpy().astype(np.float32)
    f = p.select(
        "q_id", "s1_id",
        "n_keys", "score", "max_idf", "k_name", "k_addr", "k_bigram", "k_sq", "k_pair", "k_numpair", "k_addrbag",
        "s1_name_dup", "s1_addr_dup", "src",
        pl.col("d2").cast(pl.Float32).alias("q_is_domain"),
        (pl.col("a2") == "").cast(pl.Float32).alias("q_addr_empty"),
        (pl.col("n1") == pl.col("n2")).cast(pl.Float32).alias("n_exact"),
        (pl.col("a1") == pl.col("a2")).cast(pl.Float32).alias("a_exact"),
        pl.col("n1").str.len_chars().alias("n1_len"),
        pl.col("n2").str.len_chars().alias("n2_len"),
        pl.col("a1").str.len_chars().alias("a1_len"),
        pl.col("a2").str.len_chars().alias("a2_len"),
        nums("a1").list.set_intersection(nums("a2")).list.len().alias("num_inter"),
        nums("a1").list.len().alias("num1"),
        nums("a2").list.len().alias("num2"),
        (nums("a1").list.first() == nums("a2").list.first()).cast(pl.Float32).alias("num_first_eq"),
        nums("a1").list.join(" ").alias("nums1"),
        nums("a2").list.join(" ").alias("nums2"),
        pl.col("n2").str.contains(r"[ऀ-෿]").cast(pl.Float32).alias("q_native_name"),
        pl.col("a2").str.contains(r"[ऀ-෿]").cast(pl.Float32).alias("q_native_addr"),
    )
    cols["num_ratio"] = _cp(f["nums1"].to_list(), f["nums2"].to_list(), fuzz.ratio)
    f = f.drop("nums1", "nums2")
    f = f.with_columns([pl.Series(k, v) for k, v in cols.items()])
    f = f.with_columns(
        (pl.col("n1_len") / pl.col("n2_len").clip(lower_bound=1)).cast(pl.Float32).alias("n_len_ratio"),
        (pl.col("a1_len") / pl.col("a2_len").clip(lower_bound=1)).cast(pl.Float32).alias("a_len_ratio"),
    )
    return f


def prepare_s1(s1: pl.DataFrame) -> pl.DataFrame:
    """S1-side columns needed by the feature code (cleaned name/address/squash + ambiguity counts), computed once per country."""
    amb = s1_ambiguity(s1)
    return s1.join(amb, on="id").select(
        pl.col("id").alias("s1_id"),
        pl.col("name_n").alias("n1"),
        pl.col("addr_n").alias("a1"),
        pl.col("name_sq").alias("q1"),
        "s1_name_dup",
        "s1_addr_dup",
    )


def pair_features(cand: pl.DataFrame, s1s: pl.DataFrame, queries: pl.DataFrame, chunk: int = CHUNK) -> pl.DataFrame:
    """cand: blocking output. s1s: prepare_s1(s1). Returns one feature row per candidate pair."""
    qs = queries.select(
        pl.col("id").alias("q_id"),
        pl.col("name_n").alias("n2"),
        pl.col("addr_n").alias("a2"),
        pl.col("name_sq").alias("q2"),
        pl.col("is_domain").alias("d2"),
        pl.col("native_name").cast(pl.Float32).alias("q_native_name"),
        pl.col("native_addr").cast(pl.Float32).alias("q_native_addr"),
        (pl.col("id").str.slice(1, 1) == "3").cast(pl.Float32).alias("src"),
    )
    out = []
    for i in range(0, cand.height, chunk):
        p = cand.slice(i, chunk).join(s1s, on="s1_id").join(qs, on="q_id")
        out.append(_chunk_features(p))
    return pl.concat(out)


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Names of the model input columns of a pair table (everything except ids, label, fold and score)."""
    return [c for c in df.columns if c not in ("q_id", "s1_id", "label", "fold", "p")]
