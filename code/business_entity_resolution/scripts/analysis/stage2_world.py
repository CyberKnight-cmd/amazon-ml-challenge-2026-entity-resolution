"""Stage-2 re-ranker under validation protocol v2: fit on world A, report the leaderboard metric on world B.

    --train_countries / --eval_countries  e.g. train US, evaluate India = "unseen country" (France stand-in)
Noisy-channel LLR tables are fitted out-of-fold inside the training world (4 folds by entity hash); the evaluation
world is scored with the table fitted on the whole training world, exactly as test would be.
"""
import argparse
import json
from collections import Counter

import lightgbm as lgb
import polars as pl

from entity_resolution import channel
from entity_resolution.io import WORK, load_normalized, load_truth
from entity_resolution.metrics import macro_f05
from entity_resolution.rerank import PARAMS

ap = argparse.ArgumentParser()
ap.add_argument("--train_world", default="wA")
ap.add_argument("--eval_world", default="wB")
ap.add_argument("--train_countries", nargs="+", default=["US", "India"])
ap.add_argument("--eval_countries", nargs="+", default=["US", "India"])
ap.add_argument("--rounds", type=int, default=800)
ap.add_argument("--drop", nargs="*", default=[], help="feature columns (or prefixes ending in *) to leave out")
ap.add_argument("--no_channel", action="store_true")
ap.add_argument("--save", default="", help="save model/features/table with this tag")
a = ap.parse_args()


def load(world, countries):
    """Stage-2 pair table of a world with labels and a fold id (hash of the true entity or of the query's best)."""
    df = pl.concat([pl.read_parquet(WORK / f"s2_{world}_{c}.parquet") for c in countries], how="diagonal_relaxed")
    qt = load_truth(world).rename({"other": "q_id", "s1": "true_s1"})
    df = df.join(qt, on="q_id", how="left").with_columns(
        (pl.col("s1_id") == pl.col("true_s1")).fill_null(False).cast(pl.Int8).alias("y"),
        pl.coalesce("true_s1", pl.col("s1_id").filter(pl.col("q_rank") == 0).first().over("q_id")).alias("grp"),
    )
    return df.with_columns((pl.col("grp").hash(seed=5) % 4).cast(pl.Int32).alias("f4"))


tr, ev = load(a.train_world, a.train_countries), load(a.eval_world, a.eval_countries)
print(f"train pairs={tr.height:,}  eval pairs={ev.height:,}", flush=True)
if not a.no_channel:
    keys = (tr["f4"] * 2 + tr["y"].cast(pl.Int32)).to_list()
    cnt = channel.count_events(tr["n1"].to_list(), tr["n2"].to_list(), tr["a1"].to_list(), tr["a2"].to_list(), keys)

    def table_without(k):
        pos, neg = Counter(), Counter()
        for f in range(4):
            if f != k:
                pos.update(cnt.get(f * 2 + 1, Counter()))
                neg.update(cnt.get(f * 2, Counter()))
        return channel.fit_table(pos, neg)

    tables = [table_without(k) for k in range(4)] + [table_without(-1)]
    cols = lambda d: (d["n1"].to_list(), d["n2"].to_list(), d["a1"].to_list(), d["a2"].to_list())  # noqa: E731
    tr = pl.concat([tr, channel.features(*cols(tr), tables, tr["f4"].to_list())], how="horizontal")
    ev = pl.concat([ev, channel.features(*cols(ev), tables, [4] * ev.height)], how="horizontal")

meta = {"q_id", "s1_id", "n1", "a1", "n2", "a2", "true_s1", "y", "grp", "f4"}
dropped = lambda c: any(c == d or (d.endswith("*") and c.startswith(d[:-1])) for d in a.drop)  # noqa: E731
feats = [c for c in tr.columns if c not in meta and not dropped(c)]
X = lambda d: d.select(pl.col(feats).cast(pl.Float32)).to_numpy()  # noqa: E731
fit, va = tr.filter(pl.col("f4") != 3), tr.filter(pl.col("f4") == 3)
dfit = lgb.Dataset(X(fit), fit["y"].to_numpy())
m = lgb.train(PARAMS, dfit, a.rounds, valid_sets=[lgb.Dataset(X(va), va["y"].to_numpy(), reference=dfit)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
print(f"{len(feats)} features, best iteration {m.best_iteration}", flush=True)
imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)[:15]
print("top gain:", [(f, int(g)) for g, f in imp], flush=True)
ev = ev.with_columns(pl.Series("r", m.predict(X(ev), num_iteration=m.best_iteration)))
if a.save:
    m.save_model(str(WORK / f"s2_model{a.save}.txt"), num_iteration=m.best_iteration)
    (WORK / f"s2_features{a.save}.json").write_text(json.dumps(feats))
    if not a.no_channel:
        import pickle
        pickle.dump(tables[4], open(WORK / f"s2_channel{a.save}.pkl", "wb"))

truth = load_truth(a.eval_world)
s1 = load_normalized(a.eval_world, 1)
for col, grid in (("p", (0.7,)), ("r", (0.4, 0.5, 0.6, 0.7, 0.8, 0.9))):
    best = ev.sort(["q_id", col], descending=[False, True]).group_by("q_id", maintain_order=True).first()
    for t in grid:
        tot, acc_f = 0, 0.0
        parts = []
        for c in a.eval_countries:
            ents = s1.filter(pl.col("country") == c).select(pl.col("id").alias("s1"))
            pred = best.filter(pl.col(col) >= t).select(pl.col("s1_id").alias("s1"), pl.col("q_id").alias("other")).join(ents, on="s1")
            mm = macro_f05(ents, truth.join(ents, on="s1"), pred)
            parts.append(f"{c}={mm['f05']:.5f}(P{mm['matched_precision']:.4f} R{mm['matched_recall']:.4f} S{mm['singleton_f']:.4f})")
            tot += ents.height
            acc_f += mm["f05"] * ents.height
        print(f"{'stage-1' if col == 'p' else 'stage-2'} {col}>={t}: ALL={acc_f / tot:.5f}  " + "  ".join(parts), flush=True)
