"""Build the stage-2 (re-ranker) pair table for a split: every stage-1 pair with p >= KEEP_P, with context and text
features and the raw texts needed by the noisy-channel features. Writes $ER_WORK/s2_<split>_<country>.parquet."""
import argparse
import time

import polars as pl

from entity_resolution.dataset import load_alias_table, prep_queries
from entity_resolution.features import prepare_s1
from entity_resolution.io import WORK, load_normalized
from entity_resolution.rerank import (
    add_sibling_candidates,
    base_frame,
    name_vocab,
    sibling_candidates,
    sibling_features,
    text_features,
)

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train")
ap.add_argument("--countries", nargs="+", default=["US", "India"])
ap.add_argument("--scored_tag", default="full", help="reads scored_<split>_<country>_<tag>.parquet")
ap.add_argument("--chunk", type=int, default=1_000_000)
ap.add_argument("--sibling_cands", action="store_true",
                help="add sibling-found candidate pairs (rerank.sibling_candidates); off: on the wB world they were 0.5%% "
                     "true and recovered only 0.22%% of true pairs")
a = ap.parse_args()

table = load_alias_table()
for c in a.countries:
    t0 = time.time()
    scored = pl.read_parquet(WORK / f"scored_{a.split}_{c}_{a.scored_tag}.parquet")
    s1 = load_normalized(a.split, 1).filter(pl.col("country") == c)
    rec = pl.concat([load_normalized(a.split, 2), load_normalized(a.split, 3)]).filter(pl.col("country") == c)
    if a.sibling_cands:
        sc = sibling_candidates(scored, rec)
        n_before = scored.height
        scored = add_sibling_candidates(scored, sc)
        print(f"  {c}: sibling candidates {sc.height:,} pairs, {scored.height - n_before:,} new "
              f"({scored.filter(pl.col('sc_new') == 1)['q_id'].n_unique():,} queries) ({time.time() - t0:.0f}s)", flush=True)
    q = prep_queries(rec.join(scored.select(pl.col("q_id").alias("id")).unique(), on="id"), table)
    b = base_frame(scored, prepare_s1(s1), q)
    b = b.join(sibling_features(b, s1), on=["q_id", "s1_id"], how="left")
    vocab = name_vocab(s1)
    parts = []
    for i in range(0, b.height, a.chunk):
        ch = b.slice(i, a.chunk)
        f = text_features(ch, vocab)
        keep = ch.drop("q2", "d2", "q1", "q_native_name", "q_native_addr", "src", "s1_name_dup", "s1_addr_dup")
        parts.append(pl.concat([keep, f], how="horizontal"))
        print(f"  {c}: {min(i + a.chunk, b.height):,}/{b.height:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    out = pl.concat(parts)
    out.write_parquet(WORK / f"s2_{a.split}_{c}.parquet", compression="zstd")
    print(f"[{a.split}/{c}] {out.height:,} pairs, {len(out.columns)} columns, {time.time() - t0:.0f}s", flush=True)
