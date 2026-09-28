"""End-to-end inference: per country, chunked blocking -> features -> matcher -> per-query top-2.

The S1 index is built once per country; queries (S2 and S3 records of that country) stream through in chunks so
memory stays bounded. Nothing here looks at labels or at the country name beyond partitioning."""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import polars as pl

from .blocking import BlockingConfig, build_index, query_candidates
from .blocking_v2 import build_v2_index, union_candidates
from .dataset import blocking_mode, load_alias_table, prep_queries
from .features import pair_features, prepare_s1
from .io import WORK, load_normalized


KEEP_P = 0.02


def load_model(work: Path = WORK) -> tuple[lgb.Booster, list[str]]:
    """Load the trained matcher and its exact feature-column order from $ER_WORK."""
    return lgb.Booster(model_file=str(work / "matcher.txt")), json.loads((work / "matcher_features.json").read_text())


def score_country(
    split: str,
    country: str,
    model: lgb.Booster,
    feats: list[str],
    cfg: BlockingConfig = BlockingConfig(),
    chunk_q: int = 300_000,
    limit_queries: int | None = None,
    cand_dir: Path | None = None,
    keep_p: float = KEEP_P,
    log=print,
) -> tuple[pl.DataFrame, int]:
    """Returns (scored: q_id, s1_id, p for every pair with p >= keep_p; number of queries with no candidate at all).

    Keeping all plausible candidates (not just the best) lets the decoders account for probability mass a query
    puts on entities other than its top pick."""
    s1 = load_normalized(split, 1).filter(pl.col("country") == country)
    rec = pl.concat([load_normalized(split, 2), load_normalized(split, 3)]).filter(pl.col("country") == country)
    if limit_queries:
        rec = rec.sample(min(limit_queries, rec.height), seed=7)
    table = load_alias_table()
    t0 = time.time()
    index, s1s = build_index(s1, cfg), prepare_s1(s1)
    v2 = build_v2_index(s1, table) if blocking_mode() == "v2" else None
    log(f"[{split}/{country}] S1={s1.height:,} queries={rec.height:,} blocking={blocking_mode()} index built in {time.time()-t0:.0f}s")
    out, n_cand = [], 0
    for i in range(0, rec.height, chunk_q):
        t = time.time()
        q = prep_queries(rec.slice(i, chunk_q), table)
        cand = query_candidates(index, q)
        if v2 is not None:
            cand = union_candidates(v2, q, cand)
        if cand_dir is not None:
            cand_dir.mkdir(parents=True, exist_ok=True)
            cand.select("q_id", "s1_id", "score").write_parquet(cand_dir / f"{split}_{country}_{i // chunk_q:04d}.parquet", compression="zstd")
        f = pair_features(cand, s1s, q)
        p = model.predict(f.select(pl.col(feats).cast(pl.Float32)).to_numpy())
        out.append(f.select("q_id", "s1_id").with_columns(pl.Series("p", p, dtype=pl.Float32)).filter(pl.col("p") >= keep_p))
        n_cand += cand.height
        log(f"  chunk {i // chunk_q}: {q.height:,} queries, {cand.height:,} pairs, {time.time()-t:.0f}s (total {time.time()-t0:.0f}s)")
    res = pl.concat(out)
    n_q = res["q_id"].n_unique()
    log(f"[{split}/{country}] {n_q:,}/{rec.height:,} queries have a candidate with p>={keep_p}; mean {n_cand / max(rec.height,1):.1f} cand/query")
    return res, rec.height - n_q


def assemble(all_s1: pl.DataFrame, accepted: pl.DataFrame) -> pl.DataFrame:
    """all_s1: column id in submission order. accepted: (q_id, s1_id). One row per S1 entity, '' if no match."""
    lists = accepted.group_by("s1_id").agg(pl.col("q_id").sort().str.join(",").alias("matched_entity_ids"))
    return (
        all_s1.select(pl.col("id").alias("source1_entity_id"))
        .join(lists, left_on="source1_entity_id", right_on="s1_id", how="left")
        .with_columns(pl.col("matched_entity_ids").fill_null(""))
    )


def write_submission(sub: pl.DataFrame, path: Path) -> None:
    """Write matching_results.tsv byte-for-byte in the ground-truth layout.

    Header `source1_entity_id<TAB>matched_entity_ids`; one row per S1 entity; an empty list is a truly EMPTY field.
    (polars' default CSV writer would quote empty strings as "" - which the organizers' scorer could read as a
    match id - so quoting is switched off; ids never contain tabs, quotes or newlines.)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.select("source1_entity_id", "matched_entity_ids").write_csv(path, separator="\t", quote_style="never")
