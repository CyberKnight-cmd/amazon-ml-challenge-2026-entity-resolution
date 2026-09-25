"""Train the pair matcher on train pairs and report record-level decoding quality on a held-out fold."""
import argparse
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from entity_resolution.dataset import alias_mode, cached_pair_table
from entity_resolution.features import feature_columns
from entity_resolution.io import WORK

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150_000)
ap.add_argument("--countries", nargs="+", default=["US", "India"])
ap.add_argument("--rounds", type=int, default=400)
a = ap.parse_args()

t = time.time()
df = pl.concat([cached_pair_table(c, a.n) for c in a.countries])
print(f"pairs={df.height:,} positives={df['label'].sum():,} build={time.time()-t:.0f}s", flush=True)
feats = feature_columns(df.drop("true_s1", "country"))
tr, va = df.filter(pl.col("fold") != 0), df.filter(pl.col("fold") == 0)
params = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16, max_bin=255)
dtr = lgb.Dataset(tr.select(pl.col(feats).cast(pl.Float32)).to_numpy(), tr["label"].to_numpy())
dva = lgb.Dataset(va.select(pl.col(feats).cast(pl.Float32)).to_numpy(), va["label"].to_numpy(), reference=dtr)
t = time.time()
m = lgb.train(params, dtr, a.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])
print(f"trained {m.best_iteration} rounds in {time.time()-t:.0f}s", flush=True)
m.save_model(str(WORK / "matcher.txt"))
import json
(WORK / "matcher_features.json").write_text(json.dumps(feats))
imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)[:12]
print("top gain:", [(f, int(g)) for g, f in imp])

va = va.with_columns(pl.Series("p", m.predict(va.select(pl.col(feats).cast(pl.Float32)).to_numpy())))
va.write_parquet(WORK / "val_scored.parquet")
print("alias mode:", alias_mode())
