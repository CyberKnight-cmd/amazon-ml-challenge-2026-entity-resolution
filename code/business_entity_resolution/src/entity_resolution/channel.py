"""Learned noisy-channel features: *how* a query differs from an entity, not just *how much*.

Error analysis (docs/11) showed that the dataset's decoys are built by a different corruption process than the
genuine noise. Genuine copies carry OCR-like typos ('pachec0', '5afe', 'northem' for 'northern'), dropped letters
and generic filler words ('center', 'services', 'id 10026'); decoys mutate the distinctive word in a *word-like*
way ('montora' -> 'montori', 'varshaksh' -> 'varshakshyn', 'meenaksh' -> 'meenaksha'). Plain similarity scores
both as "one character different". Here every pair is decomposed into events:

    edit operations between aligned leftover tokens   e.g. 'R|o>0|S' (replace o by 0 at the start), 'I|y|E'
    whole tokens inserted on the query side           e.g. 'center', 'services'
    whole tokens deleted from the entity side         e.g. 'electrical'

and each event gets a log-likelihood ratio learned from labelled candidate pairs (genuine vs. decoy/other). The
features are sums / minima / counts of those LLRs. Everything is learned from the provided training pairs; the
event vocabulary is character-level and language-agnostic, with character-class back-off ('R|V>V|E') so that
events never seen in training (France) still receive a sensible value.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

GROUPS = ("nop", "nins", "ndel", "aop", "ains", "adel")
_VOW = set("aeiou")
MIN_SIM = 0.5  # leftover tokens this similar are treated as one edited token, otherwise as insert + delete
MAX_TOK = 6    # pairs with more leftover tokens than this on either side are summarised by counts only


def _cls(c: str) -> str:
    """Character class used for back-off events: V(owel), C(onsonant), D(igit), N(ative script), $ (none)."""
    if c == "$":
        return "$"
    if c.isdigit():
        return "D"
    if "a" <= c <= "z":
        return "V" if c in _VOW else "C"
    return "N"


def _edit_events(a: str, b: str) -> list[str]:
    """Edit operations turning entity token `a` into query token `b`, with position (Start/Mid/End) and a class back-off."""
    out = []
    n = len(a)
    for op in Levenshtein.editops(a, b):
        i, j = op.src_pos, op.dest_pos
        ca = a[i] if i < n else "$"
        cb = b[j] if j < len(b) else "$"
        pos = "E" if i >= n - 1 else ("S" if i == 0 else "M")
        if op.tag == "replace":
            out.append(f"R|{ca}>{cb}|{pos}")
            out.append(f"R|{_cls(ca)}>{_cls(cb)}|{pos}~")
        elif op.tag == "delete":
            out.append(f"D|{ca}|{pos}")
            out.append(f"D|{_cls(ca)}|{pos}~")
        else:
            out.append(f"I|{cb}|{pos}")
            out.append(f"I|{_cls(cb)}|{pos}~")
    return out


def _field_events(s1: str, q: str) -> tuple[list[str], list[str], list[str]]:
    """(edit events, inserted query tokens, deleted entity tokens) for one field, after removing shared tokens."""
    t1, t2 = s1.split(), q.split()
    st1, st2 = set(t1), set(t2)
    l1 = [t for t in dict.fromkeys(t1) if t not in st2]
    l2 = [t for t in dict.fromkeys(t2) if t not in st1]
    if len(l1) > MAX_TOK or len(l2) > MAX_TOK:
        return [], ["#many"], ["#many"]
    ops: list[str] = []
    used = set()
    ins = []
    for b in l2:
        best, bi = 0.0, -1
        for k, a in enumerate(l1):
            if k in used:
                continue
            s = Levenshtein.normalized_similarity(a, b)
            if s > best:
                best, bi = s, k
        if bi >= 0 and best >= MIN_SIM:
            used.add(bi)
            ops.extend(_edit_events(l1[bi], b))
        else:
            ins.append(b)
    dels = [a for k, a in enumerate(l1) if k not in used]
    # a token that is all digits is described by its length, not its value (house numbers are not vocabulary)
    ins = [("#num%d" % len(t)) if t.isdigit() else t for t in ins]
    dels = [("#num%d" % len(t)) if t.isdigit() else t for t in dels]
    return ops, ins, dels


def pair_events(n1: str, n2: str, a1: str, a2: str) -> tuple[list[str], ...]:
    """All six event lists of one (entity, query) pair, in GROUPS order."""
    no, ni, nd = _field_events(n1, n2)
    ao, ai, ad = _field_events(a1, a2)
    return no, ni, nd, ao, ai, ad


def _count_chunk(args):
    """Worker: per-key presence counts of events; key = caller-supplied integer per pair (e.g. fold*2 + label)."""
    n1, n2, a1, a2, keys = args
    out: dict[int, Counter] = {}
    for x, k in zip(zip(n1, n2, a1, a2), keys):
        c = out.setdefault(int(k), Counter())
        c["#pairs"] += 1
        for gi, g in enumerate(pair_events(*x)):
            for e in set(g):
                c[(gi, e)] += 1
    return out


def _jobs(cols, chunk):
    n = len(cols[0])
    return [tuple(c[i:i + chunk] for c in cols) for i in range(0, n, chunk)]


def count_events(n1, n2, a1, a2, keys, workers: int | None = None, chunk: int = 100_000) -> dict[int, Counter]:
    """Event presence counts grouped by an integer key per pair, computed in a process pool."""
    workers = workers or min(16, os.cpu_count() or 4)
    total: dict[int, Counter] = {}
    with ProcessPoolExecutor(workers) as ex:
        for part in ex.map(_count_chunk, _jobs((n1, n2, a1, a2, list(keys)), chunk)):
            for k, c in part.items():
                total.setdefault(k, Counter()).update(c)
    return total


def fit_table(pos: Counter, neg: Counter, min_count: int = 5, alpha: float = 1.0) -> dict[str, float]:
    """LLR of every event: log P(event | genuine pair) - log P(event | wrong pair), from per-pair presence counts."""
    npos, nneg = pos["#pairs"], neg["#pairs"]
    table = {}
    for k in (set(pos) | set(neg)) - {"#pairs"}:
        p, n = pos[k], neg[k]
        if p + n < min_count:
            continue
        table[f"{k[0]}\t{k[1]}"] = math.log((p + alpha) / (npos + alpha)) - math.log((n + alpha) / (nneg + alpha))
    return table


_TABLES: list[dict[str, float]] = []


def _init(tables):
    global _TABLES
    _TABLES = tables


def _agg_chunk(args):
    """Worker: events of each pair, scored with the LLR table chosen for that pair (out-of-fold tables in training)."""
    n1, n2, a1, a2, tidx = args
    out = np.full((len(n1), len(GROUPS) * 4), np.nan, dtype=np.float32)
    for r, (x, ti) in enumerate(zip(zip(n1, n2, a1, a2), tidx)):
        table = _TABLES[ti]
        for gi, g in enumerate(pair_events(*x)):
            vals = []
            unk = 0
            for e in g:
                v = table.get(f"{gi}\t{e}")  # an unseen exact edit still has its class back-off event
                if v is None:
                    unk += 1
                else:
                    vals.append(v)
            base = gi * 4
            out[r, base + 2] = len(g)
            out[r, base + 3] = unk
            if vals:
                out[r, base] = sum(vals)
                out[r, base + 1] = min(vals)
    return out


def feature_names() -> list[str]:
    """Names of the columns produced by `features`, in order."""
    return [f"ch_{g}_{s}" for g in GROUPS for s in ("sum", "min", "cnt", "unk")]


def features(n1, n2, a1, a2, tables: list[dict[str, float]], table_idx=None, workers: int | None = None,
             chunk: int = 100_000) -> pl.DataFrame:
    """Per-pair channel features: sum / min of event LLRs, number of events and of unseen events, per event group.
    `table_idx[i]` selects which of `tables` scores pair i (default: the first for all)."""
    workers = workers or min(16, os.cpu_count() or 4)
    tidx = list(table_idx) if table_idx is not None else [0] * len(n1)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(tables,)) as ex:
        parts = list(ex.map(_agg_chunk, _jobs((n1, n2, a1, a2, tidx), chunk)))
    arr = np.vstack(parts) if parts else np.zeros((0, len(GROUPS) * 4), np.float32)
    return pl.DataFrame(arr, schema=feature_names())
