"""Apply the stage-2 re-ranker to the test set and write matching_results.tsv (with diagnostics and integrity checks).

Needs $ER_WORK/s2_test_<country>.parquet (stage2_build.py --split test) and the artefacts of
stage2_train.py --final (s2_model.txt, s2_features.json, s2_channel.pkl). Decision: each query goes to its
highest-r entity if r >= tau; everything else is left unassigned (empty lists stay empty).
"""
import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import polars as pl

from entity_resolution import channel
from entity_resolution.io import ROOT, WORK, load_normalized
from entity_resolution.pipeline import assemble, write_submission

ap = argparse.ArgumentParser()
ap.add_argument("--tau", type=float, required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--unseen_tau", type=float, default=None, help="stricter threshold for countries absent from train")
ap.add_argument("--out", default=str(ROOT / "outputs" / "submission" / "matching_results.tsv"))
a = ap.parse_args()
UNSEEN = {"France"}

model = lgb.Booster(model_file=str(WORK / f"s2_model{a.tag}.txt"))
feats = json.loads((WORK / f"s2_features{a.tag}.json").read_text())
chan = WORK / f"s2_channel{a.tag}.pkl"
tables = [pickle.load(open(chan, "rb"))] if chan.exists() else None

s1 = load_normalized("test", 1)
countries = s1["country"].unique().sort().to_list()
acc_all, best_all = [], {}
for c in countries:
    df = pl.read_parquet(WORK / f"s2_test_{c}.parquet")
    if tables is not None:
        ch = channel.features(df["n1"].to_list(), df["n2"].to_list(), df["a1"].to_list(), df["a2"].to_list(), tables)
        df = pl.concat([df, ch], how="horizontal")
    df = df.with_columns(pl.Series("r", model.predict(df.select(pl.col(feats).cast(pl.Float32)).to_numpy())))
    best = df.sort(["q_id", "r"], descending=[False, True]).group_by("q_id", maintain_order=True).first()
    tau = a.unseen_tau if (c in UNSEEN and a.unseen_tau is not None) else a.tau
    best_all[c] = best
    acc_all.append(best.filter(pl.col("r") >= tau).select("q_id", "s1_id"))
    print(f"{c}: {best.height:,} queries with a plausible pair; accepted {acc_all[-1].height:,} (tau={tau})", flush=True)

acc = pl.concat(acc_all)
out = assemble(s1.select("id"), acc)
dest = Path(a.out)
write_submission(out, dest)
raw = dest.read_bytes()
assert b'"' not in raw, "quote character in submission file (empty lists must be truly empty)"
assert raw.startswith(b"source1_entity_id\tmatched_entity_ids\n"), "unexpected header"

q_total = pl.concat([load_normalized("test", s).select("id", "country") for s in (2, 3)])
print(f"\ndecoder=stage2  written: {dest}")
for c in countries:
    n_q = q_total.filter(pl.col("country") == c).height
    n_acc = acc.join(best_all[c].select("q_id"), on="q_id").height
    o = out.join(s1.filter(pl.col("country") == c).select(pl.col("id").alias("source1_entity_id")), on="source1_entity_id")
    print(f"  {c:7s} queries={n_q:>10,} accepted={n_acc:>9,} ({n_acc / n_q:6.1%})  entities={o.height:>9,} "
          f"empty lists={(o['matched_entity_ids'] == '').mean():6.1%}")
assert out["source1_entity_id"].to_list() == s1["id"].to_list(), "row order differs from test_source1"
assert acc["q_id"].n_unique() == acc.height, "a query was assigned to more than one entity"
print("integrity checks passed: one row per S1 entity in file order; every query assigned at most once")
