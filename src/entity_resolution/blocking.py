"""Candidate generation: for every S2/S3 record ("query") find plausible S1 entities in the same country.

Every record is turned into blocking keys (rare tokens, adjacent-name-token pairs, squashed-name prefix,
pairs of rare address tokens). A query and an S1 entity become a candidate pair when they share a key that is
selective enough in S1 (document frequency <= cap). Pairs are ranked by IDF-weighted key evidence and the
top-K per query are kept. Everything is a polars join; nothing is quadratic.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class BlockingConfig:
    """Tunable blocking parameters: per-key-type frequency caps, candidates kept per query, key shapes."""
    key_cap: int = 40          # single-token keys (name / address) shared by more S1 entities are ignored
    bigram_cap: int = 150      # adjacent-name-token pairs and squashed-name prefixes are more selective
    pair_cap: int = 100        # pairs of rare address tokens
    top_k: int = 30            # candidates kept per query
    sq_prefix: int = 8         # squashed-name prefix length used as a key
    min_sq_len: int = 6
    addr_pair_tokens: int = 5  # rarest address tokens used to build pair keys (query side)
    addr_pair_tokens_s1: int = 7  # wider on the S1 side: queries may keep only S1's *common* tokens
    cross_cap: int = 0         # name-token x address-word keys ("family|cookeville"); 0 disables them


def _explode_tokens(df: pl.DataFrame, col: str, tag: str) -> pl.DataFrame:
    """One (row, key) per distinct token of `col`, keys tagged `tag:` (n = name token, a = address token)."""
    return (
        df.select("row", pl.col(col).str.split(" ").alias("t"))
        .explode("t")
        .filter(pl.col("t") != "")
        .unique()
        .select("row", (pl.lit(tag + ":") + pl.col("t")).alias("key"))
    )


def _name_bigrams(df: pl.DataFrame) -> pl.DataFrame:
    """Keys for each pair of adjacent name tokens: far more selective than single words."""
    x = (
        df.select("row", pl.col("name_n").str.split(" ").alias("t"))
        .explode("t")
        .filter(pl.col("t") != "")
        .with_columns(pl.col("t").shift(-1).over("row").alias("nxt"))
        .filter(pl.col("nxt").is_not_null())
    )
    return x.select("row", (pl.lit("b:") + pl.col("t") + pl.lit("_") + pl.col("nxt")).alias("key")).unique()


def _squash_key(df: pl.DataFrame, cfg: BlockingConfig) -> pl.DataFrame:
    """Key from the first `sq_prefix` letters of the space-free name; links glued/domain-style names to spaced ones."""
    return df.filter(pl.col("name_sq").str.len_chars() >= cfg.min_sq_len).select(
        "row", (pl.lit("q:") + pl.col("name_sq").str.slice(0, cfg.sq_prefix)).alias("key")
    )


def _cross(df: pl.DataFrame) -> pl.DataFrame:
    """(name token, address word) pairs: a common name plus a common town is still a selective combination."""
    n = df.select("row", pl.col("name_n").str.split(" ").alias("n")).explode("n").filter(pl.col("n") != "").unique()
    a = (
        df.select("row", pl.col("addr_n").str.split(" ").alias("a")).explode("a")
        .filter((pl.col("a") != "") & ~pl.col("a").str.contains(r"\d")).unique()
    )
    return n.join(a, on="row").select("row", (pl.lit("c:") + pl.col("n") + pl.lit("|") + pl.col("a")).alias("key"))


def _addr_bag(df: pl.DataFrame) -> pl.DataFrame:
    """Order-insensitive whole-address key: catches reordered but otherwise complete addresses."""
    return (
        df.select("row", pl.col("addr_n").str.split(" ").alias("t"))
        .filter(pl.col("t").list.len() >= 3)
        .select("row", (pl.lit("x:") + pl.col("t").list.sort().list.join(" ")).alias("key"))
    )


def _num_pairs(df: pl.DataFrame) -> pl.DataFrame:
    """(number token, word token) pairs: survive address truncation ('plot no 1208 ... hyderabad')."""
    x = (
        df.select("row", pl.col("addr_n").str.split(" ").alias("t"))
        .explode("t")
        .filter(pl.col("t") != "")
        .unique()
    )
    is_num = pl.col("t").str.contains(r"\d")
    nums = x.filter(is_num).rename({"t": "num"})
    words = x.filter(~is_num).rename({"t": "word"})
    return nums.join(words, on="row").select(
        "row", (pl.lit("h:") + pl.col("num") + pl.lit("|") + pl.col("word")).alias("key")
    )


def _addr_pairs(df: pl.DataFrame, tok_df: pl.DataFrame, n_tokens: int) -> pl.DataFrame:
    """Pair keys over the record's rarest address tokens that exist in S1's vocabulary."""
    x = (
        df.select("row", pl.col("addr_n").str.split(" ").alias("t"))
        .explode("t")
        .filter(pl.col("t") != "")
        .unique()
        .join(tok_df, on="t")  # keeps only tokens seen in S1, adds df
        .sort(["row", "df", "t"])
        .group_by("row", maintain_order=True)
        .head(n_tokens)
    )
    a = x.rename({"t": "t1", "df": "d1"})
    b = x.rename({"t": "t2", "df": "d2"}).select("row", "t2")
    pairs = a.join(b, on="row").filter(pl.col("t1") < pl.col("t2"))
    return pairs.select("row", (pl.lit("p:") + pl.col("t1") + pl.lit("|") + pl.col("t2")).alias("key"))


def build_keys(df: pl.DataFrame, tok_df: pl.DataFrame, cfg: BlockingConfig, n_pair_tokens: int) -> pl.DataFrame:
    """All blocking keys of a record set (token, bigram, squash, rare-address-pair, number-word, address-bag).
    
        `n_pair_tokens` = how many of the record's rarest address tokens feed the pair keys (wider on the S1 side)."""
    parts = [_cross(df)] if cfg.cross_cap else []
    return pl.concat(
        parts + [
            _explode_tokens(df, "name_n", "n"),
            _explode_tokens(df, "addr_n", "a"),
            _name_bigrams(df),
            _squash_key(df, cfg),
            _addr_pairs(df, tok_df, n_pair_tokens),
            _num_pairs(df),
            _addr_bag(df),
        ]
    )


def addr_token_df(s1: pl.DataFrame) -> pl.DataFrame:
    """Document frequency of every address token in S1 (how many entities contain it); used to find the rarest tokens."""
    return (
        s1.select("row", pl.col("addr_n").str.split(" ").alias("t"))
        .explode("t")
        .filter(pl.col("t") != "")
        .unique()
        .group_by("t")
        .agg(pl.len().alias("df"))
    )


@dataclass
class BlockIndex:
    """S1 side of blocking, built once per country and reused for every chunk of queries."""

    s1_ids: pl.DataFrame  # row, id
    tok_df: pl.DataFrame
    keys: pl.DataFrame    # (row, key, df) restricted to selective keys
    n1: int
    cfg: BlockingConfig


def build_index(s1: pl.DataFrame, cfg: BlockingConfig = BlockingConfig()) -> BlockIndex:
    """Build the reusable S1 side of blocking for one country: keys, their frequencies, with over-common keys removed."""
    s1 = s1.with_row_index("row")
    tok_df = addr_token_df(s1)
    k1 = build_keys(s1, tok_df, cfg, cfg.addr_pair_tokens_s1)
    df = k1.group_by("key").agg(pl.len().alias("df"))
    caps = {"n": cfg.key_cap, "a": cfg.key_cap, "b": cfg.bigram_cap, "q": cfg.bigram_cap, "p": cfg.pair_cap, "h": cfg.pair_cap, "x": cfg.bigram_cap, "c": cfg.cross_cap}
    k1 = (
        k1.join(df, on="key")
        .with_columns(pl.col("key").str.slice(0, 1).replace_strict(caps, return_dtype=pl.Int64).alias("cap"))
        .filter(pl.col("df") <= pl.col("cap"))
        .drop("cap")
    )
    return BlockIndex(s1.select("row", pl.col("id").alias("s1_id")), tok_df, k1, s1.height, cfg)


def query_candidates(index: BlockIndex, queries: pl.DataFrame) -> pl.DataFrame:
    """queries need columns id, name_n, addr_n, name_sq. Returns (q_id, s1_id, n_keys, score, ...) top-K per query."""
    cfg = index.cfg
    q = queries.with_row_index("row")
    kq = build_keys(q, index.tok_df, cfg, cfg.addr_pair_tokens)
    hits = (
        kq.join(index.keys, on="key", suffix="_s1")
        .with_columns(((index.n1 + 1) / pl.col("df")).log().alias("idf"), pl.col("key").str.slice(0, 1).alias("kt"))
        .group_by(["row", "row_s1"])
        .agg(
            pl.len().alias("n_keys"),
            pl.col("idf").sum().alias("score"),
            pl.col("idf").max().alias("max_idf"),
            (pl.col("kt") == "n").sum().alias("k_name"),
            (pl.col("kt") == "a").sum().alias("k_addr"),
            (pl.col("kt") == "b").sum().alias("k_bigram"),
            (pl.col("kt") == "q").sum().alias("k_sq"),
            (pl.col("kt") == "p").sum().alias("k_pair"),
            (pl.col("kt") == "h").sum().alias("k_numpair"),
            (pl.col("kt") == "x").sum().alias("k_addrbag"),
            (pl.col("kt") == "c").sum().alias("k_cross"),
        )
        .sort(["row", "score"], descending=[False, True])
        .group_by("row", maintain_order=True)
        .head(cfg.top_k)
    )
    return (
        hits.join(q.select("row", pl.col("id").alias("q_id")), on="row")
        .join(index.s1_ids.rename({"row": "row_s1"}), on="row_s1")
        .drop("row", "row_s1")
    )


def generate_candidates(s1: pl.DataFrame, queries: pl.DataFrame, cfg: BlockingConfig = BlockingConfig()) -> pl.DataFrame:
    """Convenience wrapper: build the index for `s1` and return candidates for `queries` in one call."""
    return query_candidates(build_index(s1, cfg), queries)
