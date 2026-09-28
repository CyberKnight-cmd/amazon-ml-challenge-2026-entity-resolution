"""Leaderboard-metric evaluation of a scored validation world (validation protocol v2, see build_worlds.py).

Reads $ER_WORK/scored_<world>_<country>_<tag>.parquet (q_id, s1_id, p) and reports macro F0.5 over ALL S1 entities
of the world (singletons included), per country and overall, for the plain threshold decoder over a grid of t.
    uv run python scripts/eval_world.py --world wB --tag full
"""
import argparse

import polars as pl

from entity_resolution.calibration import top2_with_ids
from entity_resolution.io import WORK, load_normalized, load_truth
from entity_resolution.metrics import macro_f05

ap = argparse.ArgumentParser()
ap.add_argument("--world", default="wB")
ap.add_argument("--countries", nargs="+", default=["US", "India"])
ap.add_argument("--tag", default="full")
ap.add_argument("--col", default="p", help="score column to threshold")
ap.add_argument("--grid", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])
a = ap.parse_args()

truth = load_truth(a.world)
s1 = load_normalized(a.world, 1)
res = {}
for c in a.countries:
    ents = s1.filter(pl.col("country") == c).select(pl.col("id").alias("s1"))
    tr = truth.join(ents, on="s1")
    best = top2_with_ids(pl.read_parquet(WORK / f"scored_{a.world}_{c}_{a.tag}.parquet").rename({a.col: "p"}) if a.col != "p"
                         else pl.read_parquet(WORK / f"scored_{a.world}_{c}_{a.tag}.parquet"))
    for t in a.grid:
        acc = best.filter(pl.col("p1") >= t)
        pred = acc.select(pl.col("s1_id").alias("s1"), pl.col("q_id").alias("other"))
        m = macro_f05(ents, tr, pred)
        res[(c, t)] = (m["f05"], ents.height)
        print(f"{a.world} {c:6s} t={t:.2f}  F0.5={m['f05']:.5f}  singleton={m['singleton_f']:.4f}  "
              f"P={m['matched_precision']:.4f}  R={m['matched_recall']:.4f}", flush=True)
for t in a.grid:
    tot = sum(res[(c, t)][1] for c in a.countries)
    print(f"{a.world} ALL    t={t:.2f}  F0.5={sum(res[(c, t)][0] * res[(c, t)][1] for c in a.countries) / tot:.5f}")
