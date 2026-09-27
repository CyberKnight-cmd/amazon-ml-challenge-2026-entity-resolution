"""Generator-aware blocking keys, used *in addition to* blocking.py (candidates = v1 top-30 + v2 top-K).

The keys invert the corruptions measured on the training pairs (docs/03, 3.9), so a noisy copy of an S1
entity lands on at least one key the entity also has:

  nb  name bag: the distinct name tokens, sorted, with up to two deleted on each side. One key family absorbs
      word reordering, duplicated words, legal-suffix swaps and one or two added/dropped/replaced/typo'd words.
  ag  anagram (sorted letters) of each bag subset with at most one deletion: glued, domain- and handle-style
      names ("opticssuperiorlovato" vs "lovato superior optics pllc").
  hn  fuzzy house number x rare address word: digit runs without leading zeros, plus every one-digit deletion
      (4211 ~ 4210 share 421; 163 ~ 63).
  cx  two rarest name tokens x rarest address words: a common name plus a street name is still selective.
  sw  first 10 letters of the squashed name x rarest address words: glued names that also dropped a word.
  ty  name tokens and their one-letter deletions: a single typo in the only informative word.

Before keys are built, digits inside words are read as letters ("5afe" -> "safe", "c0nsultancy") and tokens are
mapped to one spelling per alias class learned from the training pairs (ltd/limited, st/street/saint,
texas/tx) on BOTH sides, so equality keys see through abbreviations. Stopwords are the country's most frequent
S1 name tokens (unsupervised, so an unseen country gets its own). A key shared by more S1 entities than its
cap is ignored; candidates are ranked by the summed IDF of their shared keys.
"""

from __future__ import annotations

import itertools
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import polars as pl

FAMILIES = ("nb", "ag", "hn", "cx", "sw", "ty")
CAPS = {"nb": 200, "ag": 200, "hn": 200, "cx": 200, "sw": 200, "ty": 60}
EVIDENCE_COLS = ("v2_score", "v2_nkeys", "v2_max_idf", *(f"v2_k_{f}" for f in FAMILIES), "in_v1", "in_v2")
V1_EVIDENCE = ("n_keys", "score", "max_idf", "k_name", "k_addr", "k_bigram", "k_sq", "k_pair", "k_numpair", "k_addrbag", "k_cross")

_LEET = str.maketrans({"0": "o", "1": "l", "2": "z", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g"})
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
_DIGIT = re.compile(r"\d")
_LETTER = re.compile(r"[a-z]")


@dataclass(frozen=True)
class V2Config:
    """Tunable v2 parameters: kept candidates per query, bag deletions, S1 stopword share, query sub-chunk size."""
    top_k: int = 15
    max_del: int = 2
    stop_frac: float = 0.004
    sub_chunk: int = 50_000


def leet(t: str) -> str:
    """Read digits inside a word as the letters they imitate ('5afe' -> 'safe'); numbers and ordinals unchanged."""
    if _DIGIT.search(t) and _LETTER.search(t) and not _ORDINAL.match(t):
        return t.translate(_LEET)
    return t


def name_tokens(name: str) -> list[str]:
    """Distinct, sorted, digit-repaired name tokens."""
    return sorted({leet(t) for t in name.split() if t})


def name_keys(name: str, stop: frozenset, max_del: int) -> list[str]:
    """nb/ag/ty keys of one name: bag subsets (<= max_del deletions), their anagrams (<= 1), token typo variants."""
    toks = name_tokens(name)
    n, out = len(toks), []
    if n:
        md = min(max_del if n <= 7 else 1, n - 1)
        for d in range(md + 1):
            for comb in itertools.combinations(toks, n - d):
                if len(comb) == 1 and comb[0] in stop:
                    continue
                out.append("nb:" + " ".join(comb))
                if d <= 1:
                    out.append("ag:" + "".join(sorted("".join(comb))))
    for t in toks:
        if t in stop or not _LETTER.search(t):
            continue
        out.append("ty:" + t)
        if len(t) >= 5:
            out.extend("ty:" + t[:i] + t[i + 1:] for i in range(len(t)))
    return out


def _name_keys_chunk(args: tuple[list[str], frozenset, int]) -> list[list[str]]:
    """Worker: name keys for a chunk of distinct names."""
    names, stop, max_del = args
    return [name_keys(x, stop, max_del) for x in names]


def name_key_table(df: pl.DataFrame, stop: frozenset, max_del: int, workers: int = 14) -> pl.DataFrame:
    """(row, key) for all name-derived keys, generated once per distinct name in parallel."""
    names = df.select("name_n").unique()["name_n"].to_list()
    step = max(10_000, len(names) // (workers * 4) + 1)
    jobs = [(names[i:i + step], stop, max_del) for i in range(0, len(names), step)]
    if len(jobs) > 1:
        with ProcessPoolExecutor(workers) as ex:
            keys = [k for part in ex.map(_name_keys_chunk, jobs) for k in part]
    else:
        keys = [k for job in jobs for k in _name_keys_chunk(job)]
    kt = pl.DataFrame({"name_n": names, "key": keys}, schema={"name_n": pl.Utf8, "key": pl.List(pl.Utf8)})
    return df.select("row", "name_n").join(kt.explode("key").drop_nulls("key"), on="name_n").select("row", "key")


def alias_classes(table: pl.DataFrame | None, min_purity: float = 0.75, min_total: int = 10) -> dict[str, dict[str, str]]:
    """Union-find over learned alias pairs -> {field: {token: canonical}}. The canonical spelling of a class is its
    shortest purely alphabetic member, so 'ltd'/'limited' -> 'ltd' and 'texas'/'tx' -> 'tx' on both sides."""
    out: dict[str, dict[str, str]] = {"n": {}, "a": {}}
    if table is None:
        return out
    rank = lambda z: (not z.isalpha(), len(z), z)  # noqa: E731
    for f in ("n", "a"):
        t = table.filter(
            (pl.col("field") == f) & (pl.col("purity") >= min_purity) & (pl.col("total") >= min_total)
            & (pl.col("q_tok").str.len_chars() >= 2) & (pl.col("s_tok").str.len_chars() >= 2)
        )
        par: dict[str, str] = {}

        def find(x: str) -> str:
            while par.setdefault(x, x) != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a, b in t.select("q_tok", "s_tok").iter_rows():
            ra, rb = find(a), find(b)
            if ra != rb:
                par[max(ra, rb, key=rank)] = min(ra, rb, key=rank)
        out[f] = {x: find(x) for x in list(par) if find(x) != x}
    return out


def canonical(df: pl.DataFrame, cls: dict[str, dict[str, str]]) -> pl.DataFrame:
    """Rewrite name and address tokens to their alias-class spelling (applied identically to S1 and queries)."""
    exprs = []
    for f, col in (("n", "name_n"), ("a", "addr_n")):
        if cls.get(f):
            exprs.append(pl.col(col).str.split(" ").list.eval(pl.element().replace(cls[f])).list.join(" "))
    return df.with_columns(exprs) if exprs else df


def _addr_words(df: pl.DataFrame, wdf: pl.DataFrame, k: int) -> pl.DataFrame:
    """The k rarest non-numeric address words of each record that exist in S1's address vocabulary."""
    return (
        df.select("row", pl.col("addr_n").str.split(" ").alias("t")).explode("t")
        .filter((pl.col("t") != "") & ~pl.col("t").str.contains(r"\d")).unique()
        .join(wdf, on="t").sort(["row", "wdf", "t"]).group_by("row", maintain_order=True).head(k)
        .select("row", pl.col("t").alias("w"))
    )


def _house_numbers(df: pl.DataFrame, k: int) -> pl.DataFrame:
    """First k digit runs of the address (leading zeros removed) plus all one-digit deletions of runs >= 3 digits."""
    x = (
        df.select("row", pl.col("addr_n").str.extract_all(r"\d+").alias("n")).explode("n").drop_nulls("n")
        .with_columns(pl.col("n").str.strip_chars_start("0")).filter(pl.col("n") != "")
        .with_columns(pl.int_range(pl.len()).over("row").alias("i")).filter(pl.col("i") < k)
        .select("row", "n").unique()
    )
    parts = [x.select("row", pl.col("n").alias("v"))]
    for i in range(8):
        parts.append(x.filter(pl.col("n").str.len_chars() >= max(3, i + 1)).select(
            "row", (pl.col("n").str.slice(0, i) + pl.col("n").str.slice(i + 1)).alias("v")))
    return pl.concat(parts).unique()


def _name_rare(df: pl.DataFrame, ndf: pl.DataFrame, k: int) -> pl.DataFrame:
    """The k rarest (digit-repaired) name tokens of each record that exist in S1's name vocabulary."""
    return (
        df.select("row", pl.col("name_n").str.split(" ").alias("t")).explode("t").filter(pl.col("t") != "")
        .with_columns(pl.col("t").map_elements(leet, return_dtype=pl.Utf8)).unique()
        .join(ndf, on="t").sort(["row", "ndf", "t"]).group_by("row", maintain_order=True).head(k)
        .select("row", pl.col("t").alias("nt"))
    )


def build_keys(df: pl.DataFrame, stop: frozenset, wdf: pl.DataFrame, ndf: pl.DataFrame, cfg: V2Config, s1_side: bool) -> pl.DataFrame:
    """All v2 keys of a (canonicalized, row-indexed) record set as (row, fam, h) with h a 64-bit key hash."""
    nk = name_key_table(df, stop, cfg.max_del)
    rw = _addr_words(df, wdf, 4 if s1_side else 3)
    hn = _house_numbers(df, 3).join(rw, on="row").select("row", (pl.lit("hn:") + pl.col("v") + "|" + pl.col("w")).alias("key"))
    cx = _name_rare(df, ndf, 2).join(rw, on="row").select("row", (pl.lit("cx:") + pl.col("nt") + "|" + pl.col("w")).alias("key"))
    sq = df.select("row", pl.col("name_sq").map_elements(lambda s: leet(s)[:10], return_dtype=pl.Utf8).alias("p"))
    sw = sq.filter(pl.col("p").str.len_chars() >= 5).join(rw, on="row").select(
        "row", (pl.lit("sw:") + pl.col("p") + "|" + pl.col("w")).alias("key"))
    return pl.concat([nk, hn, cx, sw]).unique().select(
        "row", pl.col("key").str.slice(0, 2).alias("fam"), pl.col("key").hash(seed=1).alias("h"))


@dataclass
class V2Index:
    """S1 side of v2 blocking for one country, built once and reused for every chunk of queries."""
    s1_ids: pl.DataFrame
    keys: pl.DataFrame
    n1: int
    stop: frozenset
    wdf: pl.DataFrame
    ndf: pl.DataFrame
    cls: dict
    cfg: V2Config


def build_v2_index(s1: pl.DataFrame, alias_table: pl.DataFrame | None, cfg: V2Config = V2Config()) -> V2Index:
    """Canonicalize S1, derive stopwords and vocabularies, build its keys and drop keys above their family cap."""
    cls = alias_classes(alias_table)
    s1 = canonical(s1, cls).with_row_index("row")
    ndf = (
        s1.select("row", pl.col("name_n").str.split(" ").alias("t")).explode("t").filter(pl.col("t") != "")
        .with_columns(pl.col("t").map_elements(leet, return_dtype=pl.Utf8)).unique()
        .group_by("t").agg(pl.len().alias("ndf"))
    )
    wdf = (
        s1.select("row", pl.col("addr_n").str.split(" ").alias("t")).explode("t")
        .filter((pl.col("t") != "") & ~pl.col("t").str.contains(r"\d")).unique()
        .group_by("t").agg(pl.len().alias("wdf"))
    )
    stop = frozenset(ndf.filter(pl.col("ndf") >= cfg.stop_frac * s1.height)["t"].to_list())
    k = build_keys(s1, stop, wdf, ndf, cfg, s1_side=True)
    caps = pl.DataFrame({"fam": list(CAPS), "cap": list(CAPS.values())})
    k = (
        k.join(k.group_by("h").agg(pl.len().alias("df")), on="h").join(caps, on="fam")
        .filter(pl.col("df") <= pl.col("cap")).select("row", "h", "fam", "df")
    )
    return V2Index(s1.select("row", pl.col("id").alias("s1_id")), k, s1.height, stop, wdf, ndf, cls, cfg)


def v2_hits(index: V2Index, q: pl.DataFrame) -> pl.DataFrame:
    """Every (query, S1) pair sharing at least one kept v2 key, with its evidence (untruncated)."""
    q = canonical(q, index.cls).with_row_index("row")
    kq = build_keys(q, index.stop, index.wdf, index.ndf, index.cfg, s1_side=False).select("row", "h")
    hits = (
        kq.join(index.keys, on="h", suffix="_s1")
        .with_columns(((index.n1 + 1) / pl.col("df")).log().alias("idf"))
        .group_by(["row", "row_s1"])
        .agg(
            pl.col("idf").sum().alias("v2_score"),
            pl.len().alias("v2_nkeys"),
            pl.col("idf").max().alias("v2_max_idf"),
            *[(pl.col("fam") == f).sum().alias(f"v2_k_{f}") for f in FAMILIES],
        )
    )
    return (
        hits.join(q.select("row", pl.col("id").alias("q_id")), on="row")
        .join(index.s1_ids.rename({"row": "row_s1"}), on="row_s1").drop("row", "row_s1")
    )


def union_candidates(index: V2Index, q: pl.DataFrame, v1_cand: pl.DataFrame) -> pl.DataFrame:
    """v1 candidates plus the v2 top-K of each query, with both evidence sets (0 where a blocker has none).

    Processed in sub-chunks so the untruncated v2 hit table stays small."""
    k = index.cfg.top_k
    out = []
    for i in range(0, q.height, index.cfg.sub_chunk):
        qs = q.slice(i, index.cfg.sub_chunk)
        hits = v2_hits(index, qs)
        top = (
            hits.sort(["q_id", "v2_score"], descending=[False, True]).group_by("q_id", maintain_order=True).head(k)
            .select("q_id", "s1_id").with_columns(pl.lit(1, dtype=pl.Int8).alias("in_v2"))
        )
        v1 = v1_cand.join(qs.select(pl.col("id").alias("q_id")), on="q_id").with_columns(pl.lit(1, dtype=pl.Int8).alias("in_v1"))
        pairs = pl.concat([v1.select("q_id", "s1_id"), top.select("q_id", "s1_id")]).unique()
        out.append(
            pairs.join(v1, on=["q_id", "s1_id"], how="left")
            .join(top, on=["q_id", "s1_id"], how="left")
            .join(hits, on=["q_id", "s1_id"], how="left")
        )
    res = pl.concat(out, how="diagonal_relaxed")
    fill = [c for c in (*V1_EVIDENCE, *EVIDENCE_COLS) if c in res.columns]
    return res.with_columns(pl.col(c).fill_null(0) for c in fill)
