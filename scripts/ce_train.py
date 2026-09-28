"""Fine-tune the cross-encoder on hard stage-2 pairs of the train world and report a go/no-go check.

Reads s2_train_<country>.parquet from --src (default $ER_WORK; only ids, p, q_rank and the four texts are used),
keeps the uncertain pairs (stage-1 p <= HARD_P), fine-tunes on the rows selected by ce_train_rows() and compares
the cross-encoder with stage-1 p on a sample of half-B hard pairs (never used for fine-tuning).
    python scripts/ce_train.py --max_train 400000
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

from entity_resolution.cross_encoder import HARD_P, ce_train_rows, score_ce, train_ce
from entity_resolution.io import WORK, load_truth

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=None, help="folder with s2_train_*.parquet (default: $ER_WORK)")
ap.add_argument("--countries", nargs="+", default=["US", "India"])
ap.add_argument("--max_train", type=int, default=600_000)
ap.add_argument("--n_val", type=int, default=60_000)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--init", default=None, help="continue from this saved cross-encoder instead of the base model")
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--out", default="ce_model", help="output folder name inside $ER_WORK")
a = ap.parse_args()
t0 = time.time()

cols = ["q_id", "s1_id", "p", "q_rank", "n1", "a1", "n2", "a2"]
src = Path(a.src) if a.src else WORK
df = pl.concat([pl.scan_parquet(src / f"s2_train_{c}.parquet").select(cols).filter(pl.col("p") <= HARD_P).collect()
                for c in a.countries])
truth = load_truth("train").rename({"other": "q_id", "s1": "true_s1"})
df = df.join(truth, on="q_id", how="left").with_columns(
    (pl.col("s1_id") == pl.col("true_s1")).fill_null(False).cast(pl.Int8).alias("y"),
    pl.coalesce("true_s1", pl.col("s1_id").filter(pl.col("q_rank") == 0).first().over("q_id")).alias("grp"),
)
tr = df.filter(ce_train_rows())
va = df.filter(pl.col("grp").hash(seed=99) % 2 == 1)
print(f"hard pairs={df.height:,} (pos {df['y'].mean():.1%}); fine-tune rows={tr.height:,}; half-B rows={va.height:,} ({time.time()-t0:.0f}s)", flush=True)
if tr.height > a.max_train:
    tr = tr.sample(a.max_train, seed=1)
va = va.sample(min(a.n_val, va.height), seed=2)

out = WORK / a.out
kw = dict(init=a.init, seed=1) if a.init else {}
train_ce(tr.select("n1", "a1", "n2", "a2", "y"), out, epochs=a.epochs, lr=a.lr, **kw)
print(f"fine-tuned on {tr.height:,} pairs ({time.time()-t0:.0f}s)", flush=True)

ce = score_ce(out, va)
y, p = va["y"].to_numpy(), va["p"].to_numpy()
print(f"half-B hard pairs n={va.height:,} pos={y.mean():.1%}")
print(f"  stage-1 p : AUC={roc_auc_score(y, p):.4f} logloss={log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):.4f}")
print(f"  cross-enc : AUC={roc_auc_score(y, ce):.4f} logloss={log_loss(y, np.clip(ce, 1e-6, 1 - 1e-6)):.4f}")
both = (np.log(np.clip(p, 1e-6, 1)) - np.log(np.clip(1 - p, 1e-6, 1))) + (np.log(np.clip(ce, 1e-6, 1)) - np.log(np.clip(1 - ce, 1e-6, 1)))
print(f"  sum of logits (naive combination): AUC={roc_auc_score(y, both):.4f}  ({time.time()-t0:.0f}s)")
