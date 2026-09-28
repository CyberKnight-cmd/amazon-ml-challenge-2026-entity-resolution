"""Entity-level comparison of decoders on the full train world (needs scored_train_<country>_full.parquet).

Held-out protocol: S1 entities are split in two halves by hash. Everything that is *fitted* (thresholds, calibrators,
mu0) uses labels of half A only; every number reported is measured on half B entities (all their queries count,
including foreign queries wrongly assigned to them).

Decoders:  threshold  - accept a query iff p1>=t and p1-p2>=gap  (+ optional entity guard on the best record)
           expected   - calibrate, then choose per entity the k that maximizes expected F0.5
"""
import argparse
import itertools

import polars as pl

from entity_resolution.calibration import apply_calibrators, fit_calibrators, top2_with_ids
from entity_resolution.expected_f05 import decode as expected_decode
from entity_resolution.io import WORK, load_normalized, load_truth
from entity_resolution.metrics import macro_f05

ap = argparse.ArgumentParser()
ap.add_argument("--countries", nargs="+", default=["US", "India"])
ap.add_argument("--tag", default="full")
a = ap.parse_args()

scored = pl.concat([pl.read_parquet(WORK / f"scored_train_{c}_{a.tag}.parquet") for c in a.countries])
s1 = pl.concat([load_normalized("train", 1).filter(pl.col("country") == c) for c in a.countries])
half = lambda col: (pl.col(col).hash(seed=99) % 2)  # noqa: E731
truth = load_truth("train").join(s1.select(pl.col("id").alias("s1")), on="s1")
t2 = top2_with_ids(scored)
qtrue = truth.rename({"other": "q_id", "s1": "true_s1"})
print(f"entities={s1.height:,} scored queries={t2.height:,}", flush=True)

# ---- half A = fit, half B = report
ent_A = s1.filter(half("id") == 0).select(pl.col("id").alias("s1"))
ent_B = s1.filter(half("id") == 1).select(pl.col("id").alias("s1"))
truth_A, truth_B = truth.join(ent_A, on="s1"), truth.join(ent_B, on="s1")
lab = t2.join(qtrue, on="q_id", how="left").with_columns(pl.coalesce("true_s1", "q_id").alias("grp"))
fit_rows = lab.filter(half("grp") == 0)          # queries whose true entity (or, for distractors, the query) is in half A
m1, m2 = fit_calibrators(fit_rows.select(t2.columns), qtrue)
assign, ru = apply_calibrators(t2, m1, m2)


def score(acc, ents, tr):
    return macro_f05(ents, tr, acc.select(pl.col("s1_id").alias("s1"), pl.col("q_id").alias("other")).join(ents, on="s1"))


# ---- threshold decoder
rows = []
for t, gap, guard in itertools.product((0.5, 0.6, 0.7, 0.8, 0.9), (0.0, 0.1, 0.2, 0.4), (0.0, 0.9)):
    if guard and guard <= t:
        continue
    acc = t2.filter((pl.col("p1") >= t) & ((pl.col("p1") - pl.col("p2")) >= gap))
    if guard:
        acc = acc.join(acc.filter(pl.col("p1") >= guard).select("s1_id").unique(), on="s1_id", how="semi")
    rows.append(((t, gap, guard), score(acc, ent_A, truth_A)["f05"]))
(bt, bgap, bguard), fa = max(rows, key=lambda r: r[1])
acc_t = t2.filter((pl.col("p1") >= bt) & ((pl.col("p1") - pl.col("p2")) >= bgap))
if bguard:
    acc_t = acc_t.join(acc_t.filter(pl.col("p1") >= bguard).select("s1_id").unique(), on="s1_id", how="semi")
m_t = score(acc_t, ent_B, truth_B)
print(f"THRESHOLD  best on A: t={bt} gap={bgap} guard={bguard} (A f05={fa:.5f})  -> B: {m_t}", flush=True)

# ---- expected-F0.5 decoder
rows = []
for mu0 in (0.02, 0.05, 0.1, 0.2, 0.4):
    rows.append((mu0, score(expected_decode(assign, ru, mu0), ent_A, truth_A)["f05"]))
bmu, fa = max(rows, key=lambda r: r[1])
m_e = score(expected_decode(assign, ru, bmu), ent_B, truth_B)
print(f"EXPECTED   best on A: mu0={bmu} (A f05={fa:.5f})  -> B: {m_e}", flush=True)
print(f"\nGAIN on held-out entities: {m_e['f05'] - m_t['f05']:+.5f} F0.5")
m1.save_model(str(WORK / "calib_r1.txt")); m2.save_model(str(WORK / "calib_r2.txt"))
(WORK / "decode_params.json").write_text(__import__("json").dumps({"threshold": [bt, bgap, bguard], "mu0": bmu}))
