"""Normalize every raw TSV once and cache to data/interim (Parquet)."""
import time

from entity_resolution.io import load_normalized

for split in ("train", "test"):
    for s in (1, 2, 3):
        t = time.time()
        df = load_normalized(split, s, refresh=True)
        print(f"{split} s{s}: {df.height:,} rows in {time.time()-t:.0f}s", flush=True)
