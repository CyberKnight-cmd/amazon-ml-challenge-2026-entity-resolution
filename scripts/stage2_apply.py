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
ap.add_argument("--cand_out", default=str(ROOT / "outputs" / "submission" / "candidate_pairs.tsv"))
ap.add_argument("--ce", action="store_true", help="join the cross-encoder scores ce_test_<country>.parquet (model trained with --ce)")
ap.add_argument("--ce_all", action="store_true", help="with --ce: also the confident pairs' scores (model trained with --ce_all)")
a = ap.parse_args()

model = lgb.Booster(model_file=str(WORK / f"s2_model{a.tag}.txt"))
feats = json.loads((WORK / f"s2_features{a.tag}.json").read_text())
chan = WORK / f"s2_channel{a.tag}.pkl"
tables = [pickle.load(open(chan, "rb"))] if chan.exists() else None

s1 = load_normalized("test", 1)
countries = s1["country"].unique().sort().to_list()
# country is an open set: anything absent from the training S1 is "unseen" (no hard-coded names)
UNSEEN = set(countries) - set(load_normalized("train", 1)["country"].unique().to_list())
print("countries:", countries, "unseen in train:", sorted(UNSEEN))
cand_all = []
acc_all, best_all = [], {}
for c in countries:
    df = pl.read_parquet(WORK / f"s2_test_{c}.parquet")
    if a.ce:
        parts = [f"ce_test_{c}.parquet"] + ([f"ce_test_{c}_easy.parquet"] if a.ce_all else [])
        df = df.join(pl.concat([pl.read_parquet(WORK / f) for f in parts]), on=["q_id", "s1_id"], how="left")
    cand_all.append(df.select("q_id", "s1_id"))  # exactly the pairs the final model runs inference over
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

cand = pl.concat(cand_all)
csub = assemble(s1.select("id"), cand).rename({"matched_entity_ids": "candidate_entity_ids"})
Path(a.cand_out).parent.mkdir(parents=True, exist_ok=True)
csub.select("source1_entity_id", "candidate_entity_ids").write_csv(a.cand_out, separator="\t", quote_style="never")
print(f"candidates written: {a.cand_out}  pairs={cand.height:,}  per S1 entity={cand.height / s1.height:.2f}")
assert acc.join(cand, on=["q_id", "s1_id"], how="anti").height == 0, "a match that is not a candidate"

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
