"""Learn token aliases (query spelling -> S1 spelling) from matched training pairs, then rewrite queries.

For a matched (S1, query) pair we drop the tokens they share and align the *leftover* tokens in order when both
sides have the same small number of them ("pvt ltd" vs "private limited", "texas" vs "tx", a Devanagari name
vs its Latin original). Counting these alignments over many pairs and keeping the dominant partner of each
query token gives a table learned purely from the provided data. Rewriting queries with it lets exact-token
blocking keys and string features see through spelling variants and cross-script names.
"""

from __future__ import annotations

import polars as pl

MAX_LEFTOVER = 4


def _leftovers(pairs: pl.DataFrame, f: str) -> pl.DataFrame:
    """Aligned (s_tok, q_tok) rows from ordered leftover tokens of field f ('n' or 'a')."""
    base = pairs.select("row", pl.col(f + "1").str.split(" ").alias("t1"), pl.col(f + "2").str.split(" ").alias("t2"))
    one = base.select("row", "t1").explode("t1").filter(pl.col("t1") != "").with_columns(pl.int_range(pl.len()).over("row").alias("pos"))
    two = base.select("row", "t2").explode("t2").filter(pl.col("t2") != "").with_columns(pl.int_range(pl.len()).over("row").alias("pos"))
    set1 = one.select("row", pl.col("t1").alias("tok")).unique()
    set2 = two.select("row", pl.col("t2").alias("tok")).unique()
    l1 = one.join(set2, left_on=["row", "t1"], right_on=["row", "tok"], how="anti").rename({"t1": "s_tok"})
    l2 = two.join(set1, left_on=["row", "t2"], right_on=["row", "tok"], how="anti").rename({"t2": "q_tok"})
    l1 = l1.with_columns(pl.int_range(pl.len()).over("row").alias("rk"), pl.len().over("row").alias("n"))
    l2 = l2.with_columns(pl.int_range(pl.len()).over("row").alias("rk"), pl.len().over("row").alias("n"))
    return (
        l1.filter(pl.col("n") <= MAX_LEFTOVER)
        .join(l2.filter(pl.col("n") <= MAX_LEFTOVER), on=["row", "rk", "n"])
        .select("s_tok", "q_tok")
    )


def fit_aliases(pairs: pl.DataFrame, min_total: int = 5, min_purity: float = 0.6) -> pl.DataFrame:
    """pairs: columns n1,a1 (S1 side) and n2,a2 (query side), one row per matched pair.
    Returns (field, q_tok, s_tok, cnt, total, purity)."""
    pairs = pairs.with_row_index("row")
    out = []
    for f in ("n", "a"):
        al = _leftovers(pairs, f).group_by("q_tok", "s_tok").agg(pl.len().alias("cnt"))
        tot = al.group_by("q_tok").agg(pl.col("cnt").sum().alias("total"))
        best = al.sort("cnt", descending=True).group_by("q_tok", maintain_order=True).first()
        t = (
            best.join(tot, on="q_tok")
            .with_columns((pl.col("cnt") / pl.col("total")).alias("purity"), pl.lit(f).alias("field"))
            .filter((pl.col("total") >= min_total) & (pl.col("purity") >= min_purity) & (pl.col("q_tok") != pl.col("s_tok")))
        )
        out.append(t.select("field", "q_tok", "s_tok", "cnt", "total", "purity"))
    return pl.concat(out)


def apply_aliases(q: pl.DataFrame, table: pl.DataFrame, mode: str = "all") -> pl.DataFrame:
    """Rewrite name_n / addr_n of query records token by token; recompute the squashed name.
    mode: 'all' | 'native_addr' (Indic-script name tokens + all address aliases) | 'native' (Indic tokens only)."""
    res = q
    indic = pl.col("q_tok").str.contains(r"[\u0900-\u0dff]")
    for f, col in (("n", "name_n"), ("a", "addr_n")):
        t = table.filter(pl.col("field") == f)
        if mode == "native" or (mode == "native_addr" and f == "n"):
            t = t.filter(indic)
        m = dict(t.select("q_tok", "s_tok").iter_rows())
        if not m:
            continue
        res = res.with_columns(
            pl.col(col).str.split(" ").list.eval(pl.element().replace(m)).list.join(" ").alias(col)
        )
    return res.with_columns(
        pl.when(pl.col("is_domain")).then(pl.col("name_sq")).otherwise(pl.col("name_n").str.replace_all(" ", "", literal=True)).alias("name_sq")
    )


def add_native_flags(q: pl.DataFrame) -> pl.DataFrame:
    """Flag Indic-script text (before aliasing), so the matcher knows the name evidence was transliterated."""
    ind = r"[ऀ-෿]"
    return q.with_columns(
        pl.col("name_n").str.contains(ind).alias("native_name"),
        pl.col("addr_n").str.contains(ind).alias("native_addr"),
    )
