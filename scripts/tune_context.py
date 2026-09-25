"""Does the entity-context second stage beat the plain threshold? Held-out entity-level comparison.

Same protocol as tune_decode.py: entities split by hash into halves A (fit + tune) and B (report).
The context model is trained on queries whose true entity (or, for distractors, the query id) is in half A."""
import itertools
import json

import polars as pl

from entity_resolution.calibration import top2_with_ids
from entity_resolution.context import add_context, apply_context_model, fit_context_model
from entity_resolution.io import WORK, load_normalized, load_truth
from entity_resolution.metrics import macro_f05

C = ("US", "India")
scored = pl.concat([pl.read_parquet(WORK / f"scored_train_{c}_full.parquet") for c in C])
s1 = pl.concat([load_normalized("train", 1).filter(pl.col("country") == c) for c in C])
half = lambda col: (pl.col(col).hash(seed=99) % 2)  # noqa: E731
truth = load_truth("train").join(s1.select(pl.col("id").alias("s1")), on="s1")
qtrue = truth.rename({"other": "q_id", "s1": "true_s1"})
ent_A, ent_B = (s1.filter(half("id") == h).select(pl.col("id").alias("s1")) for h in (0, 1))
truth_A, truth_B = truth.join(ent_A, on="s1"), truth.join(ent_B, on="s1")

t = add_context(top2_with_ids(scored))
lab = t.join(qtrue, on="q_id", how="left").with_columns(pl.coalesce("true_s1", "q_id").alias("grp"))
model = fit_context_model(lab.filter(half("grp") == 0).select(t.columns), qtrue)
t = apply_context_model(t, model)
model.save_model(str(WORK / "context_model.txt"))


def score(acc, ents, tr):
    pred = acc.select(pl.col("s1_id").alias("s1"), pl.col("q_id").alias("other")).join(ents, on="s1")
    return macro_f05(ents, tr, pred)


rows = [(tau, score(t.filter(pl.col("r") >= tau), ent_A, truth_A)["f05"]) for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
for r in rows:
    print("A", r, flush=True)
tau, fa = max(rows, key=lambda r: r[1])
mB = score(t.filter(pl.col("r") >= tau), ent_B, truth_B)
base = score(t.filter(pl.col("p1") >= 0.7), ent_B, truth_B)
print(f"BASELINE threshold p1>=0.7           -> B f05={base['f05']:.5f}  singleton={base['singleton_f']:.4f} matched={base['matched_f']:.4f}")
print(f"CONTEXT  r>={tau} (best on A {fa:.5f}) -> B f05={mB['f05']:.5f}  singleton={mB['singleton_f']:.4f} matched={mB['matched_f']:.4f} P={mB['matched_precision']:.4f} R={mB['matched_recall']:.4f}")
print(f"GAIN {mB['f05'] - base['f05']:+.5f}")
json.dump({"tau": tau}, open(WORK / "context_params.json", "w"))
