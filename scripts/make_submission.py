"""Score the test set and write matching_results.tsv, with diagnostics and integrity checks.

Decoder parameters come from tune_decode.py (decode_params.json in $ER_WORK) unless given explicitly.
    --decoder expected   calibrate + expected-F0.5 per entity (default if tuning found it better)
    --decoder threshold  accept iff p1 >= t and p1 - p2 >= gap (+ optional entity guard)
    --decoder context    entity-context second stage (context.py): accept iff r >= tau, where r also sees how many
                         strong records the entity already has; best held-out score so far (see docs/07)
    --unseen_discount d  multiply calibrated probabilities of countries absent from train (France) by d < 1
                         to make decisions there more conservative (see docs/09_FRANCE_AND_RISKS.md)
"""
import argparse
import json

import lightgbm as lgb
import polars as pl

from entity_resolution.calibration import apply_calibrators, top2_with_ids
from entity_resolution.context import add_context, apply_context_model
from entity_resolution.expected_f05 import decode as expected_decode
from entity_resolution.io import ROOT, WORK, load_normalized
from entity_resolution.pipeline import assemble, load_model, score_country, write_submission

ap = argparse.ArgumentParser()
ap.add_argument("--decoder", choices=["expected", "threshold", "context"], default="context")
ap.add_argument("--t", type=float, default=None)
ap.add_argument("--gap", type=float, default=None)
ap.add_argument("--guard", type=float, default=None)
ap.add_argument("--mu0", type=float, default=None)
ap.add_argument("--unseen_discount", type=float, default=1.0)
ap.add_argument("--reuse", action="store_true", help="reuse scored_test_<country>_full.parquet if present")
ap.add_argument("--save_candidates", action="store_true")
ap.add_argument("--out", default=str(ROOT / "outputs" / "submission" / "matching_results.tsv"))
a = ap.parse_args()

params = json.loads((WORK / "decode_params.json").read_text()) if (WORK / "decode_params.json").exists() else {}
t, gap, guard = (params.get("threshold") or [0.7, 0.2, 0.0])
t, gap, guard = (a.t if a.t is not None else t), (a.gap if a.gap is not None else gap), (a.guard if a.guard is not None else guard)
mu0 = a.mu0 if a.mu0 is not None else params.get("mu0", 0.1)
# country is an open set: anything absent from the training S1 counts as unseen (no hard-coded names)
UNSEEN = set(load_normalized("test", 1)["country"].unique().to_list()) - set(load_normalized("train", 1)["country"].unique().to_list())

s1 = load_normalized("test", 1)
countries = s1["country"].unique().sort().to_list()
model, feats = load_model()
if a.decoder == "context":
    ctx_model = lgb.Booster(model_file=str(WORK / "context_model.txt"))
    tau = a.t if a.t is not None else json.loads((WORK / "context_params.json").read_text())["tau"]
if a.decoder == "expected":
    m1, m2 = lgb.Booster(model_file=str(WORK / "calib_r1.txt")), lgb.Booster(model_file=str(WORK / "calib_r2.txt"))

accepted, per_country_top = [], {}
for c in countries:
    path = WORK / f"scored_test_{c}_full.parquet"
    if a.reuse and path.exists():
        scored = pl.read_parquet(path)
    else:
        scored, _ = score_country("test", c, model, feats, cand_dir=(WORK / "cands") if a.save_candidates else None)
        scored.write_parquet(path, compression="zstd")
    t2 = top2_with_ids(scored)
    per_country_top[c] = t2
    if a.decoder == "context":
        r = apply_context_model(add_context(t2), ctx_model)
        if c in UNSEEN and a.unseen_discount != 1.0:
            r = r.with_columns(pl.col("r") * a.unseen_discount)
        acc = r.filter(pl.col("r") >= tau).select("q_id", "s1_id")
    elif a.decoder == "expected":
        assign, ru = apply_calibrators(t2, m1, m2)
        if c in UNSEEN and a.unseen_discount != 1.0:
            assign = assign.with_columns(pl.col("r") * a.unseen_discount)
            ru = ru.with_columns(pl.col("mass") * a.unseen_discount)
        acc = expected_decode(assign, ru, mu0)
    else:
        acc = t2.filter((pl.col("p1") >= t) & ((pl.col("p1") - pl.col("p2")) >= gap))
        if guard:
            acc = acc.join(acc.filter(pl.col("p1") >= guard).select("s1_id").unique(), on="s1_id", how="semi")
        acc = acc.select("q_id", "s1_id")
    accepted.append(acc)

acc = pl.concat(accepted)
out = assemble(s1.select("id"), acc)
dest = __import__("pathlib").Path(a.out)
write_submission(out, dest)
raw = dest.read_bytes()
assert b'"' not in raw, "quote character in submission file (empty lists must be truly empty)"
assert raw.startswith(b"source1_entity_id\tmatched_entity_ids\n"), "unexpected header"

# ---- diagnostics and integrity checks
q_total = pl.concat([load_normalized("test", s).select("id", "country") for s in (2, 3)])
print(f"\ndecoder={a.decoder}  written: {dest}")
print(f"rows: {out.height:,} (test S1 rows: {s1.height:,})")
for c in countries:
    n_q = q_total.filter(pl.col("country") == c).height
    n_acc = acc.join(per_country_top[c].select("q_id"), on="q_id").height
    ids = s1.filter(pl.col("country") == c).select(pl.col("id").alias("source1_entity_id"))
    o = out.join(ids, on="source1_entity_id")
    hi = (per_country_top[c]["p1"] >= 0.9).sum() / n_q
    print(f"  {c:7s} queries={n_q:>10,} accepted={n_acc:>9,} ({n_acc / n_q:6.1%})  entities={o.height:>9,} "
          f"empty lists={(o['matched_entity_ids'] == '').mean():6.1%}  queries with p1>=0.9: {hi:6.1%}")
assert out["source1_entity_id"].n_unique() == s1.height, "duplicate or missing S1 ids"
assert out["source1_entity_id"].to_list() == s1["id"].to_list(), "row order differs from test_source1"
assert acc["q_id"].n_unique() == acc.height, "a query was assigned to more than one entity"
print("integrity checks passed: one row per S1 entity in file order; every query assigned at most once")
