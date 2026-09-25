"""Measure blocking recall on train: full S1 index per country, random sample of S2/S3 queries."""
import argparse
import time

import polars as pl

from entity_resolution.blocking import BlockingConfig, generate_candidates
from entity_resolution.dataset import load_alias_table, prep_queries
from entity_resolution.io import load_normalized, load_truth

ap = argparse.ArgumentParser()
ap.add_argument("--country", default="US")
ap.add_argument("--aliases", type=int, default=1)
ap.add_argument("--n", type=int, default=200_000)
ap.add_argument("--key_cap", type=int, default=40)
ap.add_argument("--top_k", type=int, default=30)
ap.add_argument("--bigram_cap", type=int, default=150)
ap.add_argument("--pair_cap", type=int, default=100)
ap.add_argument("--addr_pair_tokens", type=int, default=5)
a = ap.parse_args()
cfg = BlockingConfig(key_cap=a.key_cap, top_k=a.top_k, addr_pair_tokens=a.addr_pair_tokens, bigram_cap=a.bigram_cap, pair_cap=a.pair_cap)

s1 = load_normalized("train", 1).filter(pl.col("country") == a.country)
rec = pl.concat([load_normalized("train", 2), load_normalized("train", 3)]).filter(pl.col("country") == a.country)
q = rec.sample(a.n, seed=7)
q = prep_queries(q, load_alias_table() if a.aliases else None)
truth = load_truth("train").rename({"other": "id"})

t = time.time()
cand = generate_candidates(s1, q, cfg)
dt = time.time() - t

q = q.join(truth, on="id", how="left")  # s1 = true entity, null for distractors
matched = q.filter(pl.col("s1").is_not_null())
hit = cand.select("q_id", "s1_id").join(matched.select(pl.col("id").alias("q_id"), pl.col("s1").alias("s1_id")), on=["q_id", "s1_id"], how="inner")
found = hit.select("q_id").unique().height
per_q = cand.group_by("q_id").len()["len"]
print(f"{a.country}: S1={s1.height:,} queries={q.height:,} matched={matched.height:,} time={dt:.0f}s")
print(f"  candidates/query mean={cand.height / q.height:.1f}  queries with >=1 cand={per_q.len() / q.height:.3f}")
print(f"  RECALL (true S1 in candidates) = {found / matched.height:.4f}")

miss = matched.join(hit.select(pl.col("q_id").alias("id")).unique(), on="id", how="anti")
native = pl.col("name_n").str.contains(r"[ऀ-෿]")
print(f"  missed={miss.height:,}: native-name={miss.filter(native).height:,} domain={miss.filter(pl.col('is_domain')).height:,} empty-addr={miss.filter(pl.col('addr_n') == '').height:,}")
print(f"  recall native-name = {1 - miss.filter(native).height / max(matched.filter(native).height, 1):.4f}  domain = {1 - miss.filter(pl.col('is_domain')).height / max(matched.filter(pl.col('is_domain')).height, 1):.4f}  empty-addr = {1 - miss.filter(pl.col('addr_n') == '').height / max(matched.filter(pl.col('addr_n') == '').height, 1):.4f}")
s1i = s1.select(pl.col("id").alias("s1"), pl.col("name_n").alias("n1"), pl.col("addr_n").alias("a1"))
for r in miss.join(s1i, on="s1").head(8).iter_rows(named=True):
    print("   MISS:", r["n1"], "|", r["a1"], "  <>  ", r["name_n"], "|", r["addr_n"])

if a.country == "India":
    nm = miss.filter(native).join(s1i, on="s1")
    print("  -- native-name misses (sample)")
    for r in nm.sample(12, seed=2).iter_rows(named=True):
        print("   NMISS:", r["n1"], "|", r["a1"], "  <>  ", r["name_n"], "|", r["addr_n"])
    lat = nm.filter(pl.col("addr_n").str.contains(r"[a-z0-9]"))
    print(f"  native-name misses with any latin/digit in addr: {lat.height}/{nm.height}; addr fully empty: {nm.filter(pl.col('addr_n')=='').height}")

other = miss.filter(~native & (pl.col("addr_n") != "")).join(s1i, on="s1")
print(f"  -- other misses (latin name, non-empty addr): {other.height}")
for r in other.sample(min(14, other.height), seed=5).iter_rows(named=True):
    print("   OMISS:", r["n1"], "|", r["a1"], "  <>  ", r["name_n"], "|", r["addr_n"])
