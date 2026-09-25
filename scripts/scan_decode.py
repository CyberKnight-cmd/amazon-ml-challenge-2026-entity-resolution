"""Record-level decoding scan on the held-out fold (val_scored.parquet from train_model.py)."""
import polars as pl

from entity_resolution.decode import accept, top2
from entity_resolution.io import WORK

va = pl.read_parquet(WORK / "val_scored.parquet")
meta = pl.concat([pl.read_parquet(f) for f in sorted(WORK.glob("meta_*_300000_7.parquet"))])
qs = meta.filter(pl.col("fold") == 0).select("q_id", "true_s1", "country")  # ALL fold-0 queries, incl. those with no candidates
n_matched = qs.filter(pl.col("true_s1").is_not_null()).height
n_dist = qs.height - n_matched
print(f"val queries={qs.height:,} matched={n_matched:,} distractors={n_dist:,}")
t2 = top2(va).join(qs, on="q_id")
rows = []
for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
    for gap in (0.0, 0.2, 0.4):
        acc = accept(t2, t, gap).join(qs, on="q_id")
        tp = acc.filter(pl.col("s1_id") == pl.col("true_s1")).height
        fp_wrong = acc.filter(pl.col("true_s1").is_not_null() & (pl.col("s1_id") != pl.col("true_s1"))).height
        fp_dist = acc.filter(pl.col("true_s1").is_null()).height
        prec = tp / max(acc.height, 1)
        rec = tp / n_matched
        f05 = 1.25 * prec * rec / (0.25 * prec + rec) if tp else 0
        rows.append((t, gap, acc.height, round(prec, 4), round(rec, 4), round(f05, 4), fp_wrong, fp_dist))
pl.Config.set_tbl_rows(40)
print(pl.DataFrame(rows, schema=["t", "gap", "accepted", "precision", "recall", "f05", "fp_wrong_entity", "fp_distractor"], orient="row"))
best = max(rows, key=lambda r: r[5]); print("best record-level F0.5:", best)
for c in ("US", "India"):
    q = t2.filter(pl.col("country") == c)
    acc = accept(q, best[0], best[1]).join(qs, on="q_id")
    m = q.filter(pl.col("true_s1").is_not_null()).height
    tp = acc.filter(pl.col("s1_id") == pl.col("true_s1")).height
    print(f"  {c}: precision={tp/max(acc.height,1):.4f} recall={tp/m:.4f}")
