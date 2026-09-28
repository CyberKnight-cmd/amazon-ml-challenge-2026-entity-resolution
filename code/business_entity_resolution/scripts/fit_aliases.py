"""Fit the alias table on matched train pairs from folds != 0 (fold 0 stays clean for validation)."""
import time

import polars as pl

from entity_resolution.aliases import fit_aliases
from entity_resolution.dataset import N_FOLDS
from entity_resolution.io import WORK, load_normalized, load_truth

t = time.time()
s1 = load_normalized("train", 1).select(pl.col("id").alias("s1"), pl.col("name_n").alias("n1"), pl.col("addr_n").alias("a1"))
rec = pl.concat([load_normalized("train", 2), load_normalized("train", 3)]).select(pl.col("id").alias("other"), pl.col("name_n").alias("n2"), pl.col("addr_n").alias("a2"))
gt = load_truth("train").filter(pl.col("s1").hash(seed=13) % N_FOLDS != 0).sample(1_500_000, seed=3)
pairs = gt.join(s1, on="s1").join(rec, on="other")
table = fit_aliases(pairs)
WORK.mkdir(parents=True, exist_ok=True)
table.write_parquet(WORK / "aliases.parquet")
print(f"{pairs.height:,} pairs -> {table.height:,} aliases in {time.time()-t:.0f}s")
print(table.group_by("field").len())
print("name aliases (top by count):"); print(table.filter(pl.col("field") == "n").sort("cnt", descending=True).head(25))
print("addr aliases (top by count):"); print(table.filter(pl.col("field") == "a").sort("cnt", descending=True).head(25))
