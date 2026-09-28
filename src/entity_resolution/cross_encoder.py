"""Cross-encoder for the hard pairs: a small multilingual transformer that reads both records at once.

The string features of stages 1-2 compare the two records through fixed similarity functions. A cross-encoder
instead reads "S1 name | S1 address" and "query name | query address" together and learns, from the training
labels, which differences are noise (a transliterated name, a typo, a truncated address) and which are a
different business (a swapped word, another house number). It is only run on the uncertain pairs (stage-1
p <= HARD_P), where those distinctions decide the outcome; its probability becomes one stage-2 feature.

Model: intfloat/multilingual-e5-small (MIT licence, 118 M parameters; covers the Indic scripts and French),
fine-tuned as a single-logit pair classifier. Only the provided training data is used for fine-tuning.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import polars as pl

BASE_MODEL = "intfloat/multilingual-e5-small"
HARD_P = 0.98
MAX_LEN = 96


def ce_train_rows(grp: str = "grp") -> pl.Expr:
    """Rows whose group (true entity, or a distractor's best candidate) the cross-encoder is fine-tuned on: a third of
    stage-2's fitting half A (half = hash(seed=99) % 2 == 0, as in stage2_train.py). Stage 2 leaves these rows out,
    so no stage-2 row ever carries a cross-encoder score fitted on its own label, and half B stays untouched."""
    return ((pl.col(grp).hash(seed=77) % 3) == 0) & ((pl.col(grp).hash(seed=99) % 2) == 0)


def pair_texts(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    """'name | address' of the S1 side and of the query side (normalized texts n1,a1 / n2,a2)."""
    a = df.select((pl.col("n1") + " | " + pl.col("a1")).alias("x"))["x"].to_list()
    b = df.select((pl.col("n2") + " | " + pl.col("a2")).alias("x"))["x"].to_list()
    return a, b


def train_ce(df: pl.DataFrame, out_dir: Path, epochs: int = 1, batch: int = 128, lr: float = 5e-5, log=print,
             init: str | Path = BASE_MODEL, seed: int = 0) -> None:
    """Fine-tune `init` (BASE_MODEL, or a saved cross-encoder to continue) as a pair classifier on df (n1,a1,n2,a2,y);
    save model + tokenizer to out_dir."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(init)
    model = AutoModelForSequenceClassification.from_pretrained(init, num_labels=1).to(dev)
    a, b = pair_texts(df)
    y = df["y"].cast(pl.Float32).to_numpy()
    n = len(a)
    steps = epochs * math.ceil(n / batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    model.train()
    t0, step = time.time(), 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            enc = tok([a[j] for j in idx], [b[j] for j in idx], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), torch.from_numpy(y[idx]).to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0:
                log(f"  step {step}/{steps} loss={loss.item():.4f} ({(i + batch) / (time.time() - t0):.0f} pairs/s)")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def score_ce(model_dir: Path, df: pl.DataFrame, batch: int = 1024, log=print) -> np.ndarray:
    """Match probability of every pair of df (n1,a1,n2,a2) under the fine-tuned cross-encoder."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).half().eval()
    a, b = pair_texts(df)
    # sort by length so each batch pads little; restore order at the end
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])
    out = np.empty(len(a), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for k, i in enumerate(range(0, len(a), batch)):
            idx = order[i:i + batch]
            enc = tok([a[j] for j in idx], [b[j] for j in idx], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(dev)
            out[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
            if k % 200 == 0 and k:
                log(f"  scored {i + batch:,}/{len(a):,} ({(i + batch) / (time.time() - t0):.0f} pairs/s)")
    return out
