"""Score every query of a split with the trained matcher; writes top-2 tables to WORK."""
import argparse

from entity_resolution.blocking import BlockingConfig
from entity_resolution.io import WORK
from entity_resolution.pipeline import load_model, score_country

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train")
ap.add_argument("--countries", nargs="+", default=["US", "India"])
ap.add_argument("--limit_queries", type=int, default=None)
ap.add_argument("--chunk", type=int, default=300_000)
ap.add_argument("--save_candidates", action="store_true")
ap.add_argument("--cross_cap", type=int, default=0, help="enable name x address-word blocking keys with this cap")
ap.add_argument("--keep_p", type=float, default=0.02, help="keep every pair with stage-1 p >= keep_p")
ap.add_argument("--tag", default=None, help="output tag (default: 'full' or the query limit)")
a = ap.parse_args()

model, feats = load_model()
for c in a.countries:
    res, missing = score_country(a.split, c, model, feats, chunk_q=a.chunk, limit_queries=a.limit_queries,
                                 cand_dir=(WORK / "cands") if a.save_candidates else None,
                                 cfg=BlockingConfig(cross_cap=a.cross_cap), keep_p=a.keep_p)
    tag = a.tag or (f"{a.limit_queries}" if a.limit_queries else "full")
    res.write_parquet(WORK / f"scored_{a.split}_{c}_{tag}.parquet", compression="zstd")
    print(f"saved scored pairs for {c}: {res.height:,} rows; {missing:,} queries without a plausible candidate", flush=True)
