"""Build test-like validation worlds from train (see docs/08, "validation protocol v2").

Measured on the data (26 Sep): the test set has ~5.8 S2/S3 records per S1 entity vs ~4.7 in train, and the extra
~19% are clean records (no empty addresses, no website-style names: the test rates of both are exactly the train
rates x 4.67/5.76) of businesses absent from S1 - 35% of test US/India records have no plausible candidate vs 23%
in train. Each train country is also about twice the size of its test counterpart (US 1.32M vs 0.66M S1).

So each country's S1 entities are split by hash into two halves, and each half becomes a self-contained world:
    S1      = the half's entities
    records = all S2/S3 records matched to those entities
            + the unmatched train records assigned to the half by hash
            + "orphans": clean records (non-empty address, not website-style) matched to the OTHER half's entities,
              sampled so that the world has the test's records-per-entity ratio.
World A is used for fitting, world B only for reporting. They are written as pseudo-splits 'wA' / 'wB' with the
same file layout as data/interim/<split>_s<k>.parquet plus <split>_truth.parquet, so every script runs unchanged.
"""
import argparse

import polars as pl

from entity_resolution.io import INTERIM, load_normalized, load_truth

ap = argparse.ArgumentParser()
ap.add_argument("--target_ratio", type=float, default=5.8, help="S2/S3 records per S1 entity in each world (test: 5.8)")
ap.add_argument("--seed", type=int, default=2026)
a = ap.parse_args()

s1 = load_normalized("train", 1)
q = pl.concat([load_normalized("train", 2), load_normalized("train", 3)])
truth = load_truth("train")  # s1, other
h = lambda col: (pl.col(col).hash(seed=a.seed) % 2)  # noqa: E731
s1 = s1.with_columns(h("id").alias("w"))
owner = truth.join(s1.select(pl.col("id").alias("s1"), "w"), on="s1").select(pl.col("other").alias("id"), "s1", "w")
q = q.join(owner, on="id", how="left").with_columns(
    pl.coalesce("w", h("id")).alias("w"), pl.col("s1").is_not_null().alias("matched")
)
clean = (pl.col("addr_n") != "") & ~pl.col("is_domain")

for w, name in ((0, "wA"), (1, "wB")):
    ents = s1.filter(pl.col("w") == w)
    own = q.filter(pl.col("w") == w)
    parts = []
    for c in ents["country"].unique().to_list():
        n_s1 = ents.filter(pl.col("country") == c).height
        own_c = own.filter(pl.col("country") == c)
        need = max(0, int(a.target_ratio * n_s1) - own_c.height)
        pool = q.filter((pl.col("w") != w) & pl.col("matched") & clean & (pl.col("country") == c))
        orph = pool.sample(min(need, pool.height), seed=a.seed + w).with_columns(pl.lit(None, dtype=pl.Utf8).alias("s1"))
        parts += [own_c, orph]
        print(f"{name} {c}: S1={n_s1:,} own records={own_c.height:,} orphans={orph.height:,} "
              f"-> {(own_c.height + orph.height) / n_s1:.2f} per entity, matched share {own_c['matched'].sum() / (own_c.height + orph.height):.1%}")
    recs = pl.concat(parts)
    cols = [c for c in load_normalized("train", 2).columns]
    ents.select(cols).write_parquet(INTERIM / f"{name}_s1.parquet")
    for k in (2, 3):
        recs.filter(pl.col("id").str.starts_with(f"S{k}-")).select(cols).write_parquet(INTERIM / f"{name}_s{k}.parquet")
    recs.filter(pl.col("s1").is_not_null()).select("s1", pl.col("id").alias("other")).write_parquet(INTERIM / f"{name}_truth.parquet")
    print(f"{name}: written ({ents.height:,} S1, {recs.height:,} records)")
