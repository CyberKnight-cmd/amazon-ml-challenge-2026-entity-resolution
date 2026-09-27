"""Score the hard pairs (stage-1 p <= HARD_P) of stage-2 tables with the fine-tuned cross-encoder.

Reads $ER_WORK/s2_<split>_<country>.parquet and $ER_WORK/ce_model, writes $ER_WORK/ce_<split>_<country>.parquet
(q_id, s1_id, ce). Pairs above HARD_P get no score (a missing value for stage 2).
    uv run python scripts/ce_score.py --split test --countries France India US
"""
import argparse
import time

import polars as pl

from entity_resolution.cross_encoder import HARD_P, score_ce
from entity_resolution.io import WORK

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train")
ap.add_argument("--countries", nargs="+", default=["US", "India"])
a = ap.parse_args()

for c in a.countries:
    t0 = time.time()
    df = pl.scan_parquet(WORK / f"s2_{a.split}_{c}.parquet").select("q_id", "s1_id", "p", "n1", "a1", "n2", "a2").filter(
        pl.col("p") <= HARD_P).collect()
    ce = score_ce(WORK / "ce_model", df)
    df.select("q_id", "s1_id").with_columns(pl.Series("ce", ce)).write_parquet(WORK / f"ce_{a.split}_{c}.parquet")
    print(f"[{a.split}/{c}] scored {df.height:,} hard pairs in {time.time() - t0:.0f}s", flush=True)
