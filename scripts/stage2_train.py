"""Train and evaluate the stage-2 re-ranker on the full train world (entity-level, held-out entities).

Protocol (same halves as tune_context.py): entities split by hash into A (fit) and B (report). Noisy-channel LLR
tables are fitted out-of-fold inside A (4 folds by entity hash) so that training rows never see a table fitted on
their own labels; B rows are scored with the table fitted on all of A. With --final the same is done on the whole
train world and the artefacts for test inference are saved (s2_model.txt, s2_features.json, s2_channel.pkl).
"""
import argparse
import json
import pickle
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from entity_resolution import channel
from entity_resolution.io import WORK, load_truth
from entity_resolution.metrics import macro_f05
from entity_resolution.rerank import PARAMS

ap = argparse.ArgumentParser()
ap.add_argument("--countries", nargs="+", default=["US", "India"])
ap.add_argument("--rounds", type=int, default=2000)
ap.add_argument("--final", action="store_true", help="train on everything and save artefacts for test")
ap.add_argument("--final_rounds", type=int, default=0, help="rounds for --final (default: best iteration of eval run)")
ap.add_argument("--no_channel", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--drop", nargs="*", default=[], help="feature columns (or prefixes ending in *) to leave out, for ablations")
ap.add_argument("--no_sc", action="store_true", help="ablation: remove sibling-found pairs (sc_new=1) and the sc_* features")
ap.add_argument("--ce", action="store_true", help="add the cross-encoder score (ce_train_<country>.parquet) as a feature and "
                "leave out the rows the cross-encoder was fine-tuned on")
a = ap.parse_args()
t0 = time.time()

df = pl.concat([pl.read_parquet(WORK / f"s2_train_{c}.parquet") for c in a.countries], how="diagonal_relaxed")
if a.no_sc and "sc_new" in df.columns:
    df = df.filter(pl.col("sc_new") == 0)
    a.drop = [*a.drop, "sc_*"]
truth = load_truth("train")
qtrue = truth.rename({"other": "q_id", "s1": "true_s1"})
half = lambda col, seed=99: (pl.col(col).hash(seed=seed) % 2)  # noqa: E731
df = df.join(qtrue, on="q_id", how="left").with_columns(
    (pl.col("s1_id") == pl.col("true_s1")).fill_null(False).cast(pl.Int8).alias("y"),
    # a distractor travels with the entity it most likely attaches to, so A/B never share an entity's decoys
    pl.coalesce("true_s1", pl.col("s1_id").filter(pl.col("q_rank") == 0).first().over("q_id")).alias("grp"),
)
df = df.with_columns(half("grp").alias("half"), (pl.col("grp").hash(seed=5) % 4).cast(pl.Int32).alias("f4"))
if a.ce:
    from entity_resolution.cross_encoder import ce_train_rows

    ce = pl.concat([pl.read_parquet(WORK / f"ce_train_{c}.parquet") for c in a.countries])
    n0 = df.height
    df = df.filter(~ce_train_rows()).join(ce, on=["q_id", "s1_id"], how="left")
    print(f"cross-encoder: {ce.height:,} scored pairs; left out {n0 - df.height:,} fine-tuning rows", flush=True)
print(f"pairs={df.height:,} positives={df['y'].sum():,} ({time.time() - t0:.0f}s)", flush=True)

fit_rows = df if a.final else df.filter(pl.col("half") == 0)
if not a.no_channel:
    keys = (fit_rows["f4"] * 2 + fit_rows["y"].cast(pl.Int32)).to_list()
    cnt = channel.count_events(fit_rows["n1"].to_list(), fit_rows["n2"].to_list(), fit_rows["a1"].to_list(),
                               fit_rows["a2"].to_list(), keys)
    from collections import Counter

    def table_without(k):
        pos, neg = Counter(), Counter()
        for f in range(4):
            if f != k:
                pos.update(cnt.get(f * 2 + 1, Counter()))
                neg.update(cnt.get(f * 2, Counter()))
        return channel.fit_table(pos, neg)

    tables = [table_without(k) for k in range(4)] + [table_without(-1)]
    print(f"channel tables: {[len(t) for t in tables]} events ({time.time() - t0:.0f}s)", flush=True)
    # rows used for fitting get their out-of-fold table; the others (B) the table fitted on all fitting rows
    in_fit = pl.lit(True) if a.final else (pl.col("half") == 0)
    tidx = df.select(pl.when(in_fit).then(pl.col("f4")).otherwise(4))[:, 0].to_list()
    ch = channel.features(df["n1"].to_list(), df["n2"].to_list(), df["a1"].to_list(), df["a2"].to_list(), tables, tidx)
    df = pl.concat([df, ch], how="horizontal")
    print(f"channel features done ({time.time() - t0:.0f}s)", flush=True)

drop = {"q_id", "s1_id", "n1", "a1", "n2", "a2", "true_s1", "y", "grp", "half", "f4"}
feats = [c for c in df.columns if c not in drop and not any(c == x or (x.endswith("*") and c.startswith(x[:-1])) for x in a.drop)]
print(f"{len(feats)} features" + (f" (dropped {a.drop})" if a.drop else ""), flush=True)
X = lambda d: d.select(pl.col(feats).cast(pl.Float32)).to_numpy()  # noqa: E731

if a.final:
    rounds = a.final_rounds or json.loads((WORK / f"s2_eval{a.tag}.json").read_text())["best_iter"]
    m = lgb.train(PARAMS, lgb.Dataset(X(df), df["y"].to_numpy()), rounds)
    m.save_model(str(WORK / f"s2_model{a.tag}.txt"))
    (WORK / f"s2_features{a.tag}.json").write_text(json.dumps(feats))
    if not a.no_channel:
        pickle.dump(tables[4], open(WORK / f"s2_channel{a.tag}.pkl", "wb"))
    print(f"final model saved: {rounds} rounds, {len(feats)} features ({time.time() - t0:.0f}s)")
    raise SystemExit

A, B = df.filter(pl.col("half") == 0), df.filter(pl.col("half") == 1)
tr, va = A.filter(pl.col("f4") != 3), A.filter(pl.col("f4") == 3)
dtr = lgb.Dataset(X(tr), tr["y"].to_numpy())
dva = lgb.Dataset(X(va), va["y"].to_numpy(), reference=dtr)
m = lgb.train(PARAMS, dtr, a.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
print(f"trained {m.best_iteration} rounds ({time.time() - t0:.0f}s)", flush=True)
imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)[:25]
print("top gain:", [(f, int(g)) for g, f in imp])
json.dump({"best_iter": m.best_iteration}, open(WORK / f"s2_eval{a.tag}.json", "w"))

B = B.with_columns(pl.Series("r", m.predict(X(B))))
va = va.with_columns(pl.Series("r", m.predict(X(va))))
B.select("q_id", "s1_id", "p", "r", "y").write_parquet(WORK / f"s2_predB{a.tag}.parquet")

from entity_resolution.io import load_normalized  # noqa: E402

ents = load_normalized("train", 1).filter(pl.col("country").is_in(a.countries)).select(pl.col("id").alias("s1"))
entB = ents.filter(half("s1") == 1)
truthB = truth.join(entB, on="s1")


def score(pairs, col, tau):
    best = pairs.sort(["q_id", col], descending=[False, True]).group_by("q_id", maintain_order=True).first()
    acc = best.filter(pl.col(col) >= tau)
    pred = acc.select(pl.col("s1_id").alias("s1"), pl.col("q_id").alias("other")).join(entB, on="s1")
    return macro_f05(entB, truthB, pred)


base = score(B, "p", 0.7)
print(f"B stage-1 p>=0.7: f05={base['f05']:.5f} singleton={base['singleton_f']:.4f} P={base['matched_precision']:.4f} R={base['matched_recall']:.4f}")
for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    r = score(B, "r", tau)
    print(f"B stage-2 r>={tau}: f05={r['f05']:.5f} singleton={r['singleton_f']:.4f} P={r['matched_precision']:.4f} R={r['matched_recall']:.4f}", flush=True)
