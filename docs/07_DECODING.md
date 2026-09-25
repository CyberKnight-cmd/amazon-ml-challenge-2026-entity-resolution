# 07 — Decoding: from probabilities to the final lists (including the empty ones)

**Code:** `decode.py` (simple threshold), `calibration.py`, `expected_f05.py` · **Script:**
`scripts/tune_decode.py` · **Tests:** `tests/test_expected_f05.py`, `tests/test_pipeline_pieces.py`

## 7.1 The problem this solves

The matcher (06) gives every (query, entity) pair a probability. The leaderboard, however, wants **one list per
entity**. Turning probabilities into lists sounds trivial ("keep p > 0.5") but is where a lot of score is won or
lost, for three reasons that come straight from the problem statement:

1. **A query has at most one owner.** If a query gives 0.9 to entity A and 0.88 to entity B, it is *not* safe to
   assign it to either; at most one is right and we do not know which.
2. **Precision counts double.** A wrong record hurts more than a missed one.
3. **Singletons are all-or-nothing.** For an entity with no true match, predicting *one* record scores 0;
   predicting nothing scores 1 (a 1-point swing on that entity). So the decision "predict nothing" has to be
   made deliberately, entity by entity.

## 7.2 Decoder A — the threshold baseline (`decode.py`)

For each query take its best candidate (probability `p1`) and its runner-up (`p2`). Accept the query into its best
entity if

* `p1 ≥ t` (confident enough), and
* `p1 − p2 ≥ gap` (clearly better than the alternative).

Then each entity's list = the accepted queries pointing at it; entities with none get an empty list. A third,
optional rule, the **guard**, drops an entity's whole list unless at least one of its records has `p1 ≥ guard`
(protection for singletons). `t`, `gap`, `guard` are tuned on the leaderboard-metric emulation (08).

**Strengths:** simple, robust, explainable. **Weakness:** one global `t` treats a lone weak record and a record
inside a strong cluster identically, although the score impact differs sharply.

## 7.3 Decoder B — expected-F0.5 (`calibration.py` + `expected_f05.py`)

The idea: since we know the exact scoring rule, **choose, for each entity, the answer with the highest expected
score** — and "predict nothing" is just one of the options.

### Step 1 — Calibration (`calibration.py`)
The matcher scores pairs in isolation; two things distort its probabilities:

* **Competition:** `p1 = 0.9` with `p2 = 0.88` is far less certain than `p1 = 0.9` with `p2 = 0.02`.
* **Selection:** the best of ≈ 28 candidates is optimistic compared with a random one.

So we fit two small monotone LightGBM models on labelled train data:

* `r1` = P(the query's best candidate is its true entity | `p1`, `p2`)
* `r2` = P(the query's runner-up is its true entity | `p1`, `p2`)

`r1` becomes the probability that a record belongs to its entity; `r2`, summed per entity, becomes the probability
mass that "leaks" to an entity through *other* queries' second choices (it estimates true matches we did not see
as first choices).

### Step 2 — The expected-score maths (`expected_f05.py`)
For one entity, let its candidate records be `i = 1..M` (queries whose best entity it is), with probabilities
`r_i`, sorted high to low. The best answer is always a **prefix** ("the top k records") — the standard result
for F-measure optimization — so we only have to choose `k = 0, 1, …, M`.

Let `TP` = correct records among the k chosen, `N` = total number of true matches of the entity (including ones
we never saw, modelled as a small Poisson count with mean `mass + mu0`). The exact per-entity score is

```
k ≥ 1 :  F0.5 = 1.25 · TP / (0.25 · N + k)        # algebraically equal to 1.25·P·R / (0.25·P + R)
k = 0 :  F0.5 = 1 if N = 0 else 0                 # the singleton rule
```

Because `TP` and `N − TP` are independent sums of Bernoulli variables, their distributions are computed exactly
by dynamic programming ("Poisson-binomial"), and `E[F0.5]` is evaluated for every `k` in one vectorized pass over
all entities. We pick the `k` with the highest expectation (ties → smaller `k`, i.e. more conservative).

**What emerges automatically:**

* A lone record with `r = 0.30` → predict nothing (expected score of nothing ≈ 0.7, of guessing ≈ 0.3).
* A lone record with `r = 0.98` → include it.
* Two records at 0.95 and 0.90 plus a doubtful 0.20 → include the first two, drop the third.
* Precision-tilt and singleton protection come from the *formula*, not from hand-tuned guards.

`tests/test_expected_f05.py` verifies the dynamic program against brute-force enumeration of every outcome, and
checks the three behaviours above.

### Step 3 — One free parameter
`mu0`: the prior expected number of an entity's true matches that we did not see at all (blocking misses etc.).
It is tuned on held-out entities.

## 7.3b Decoder C — entity-context second stage (`context.py`) — the one we submit

### Why it exists (measured, not guessed)
We broke the entity-level loss of the threshold decoder into failure types on the full train world (see 08):

| Failure type | Loss (of a 1.0 scale) | Share of total loss |
|---|---|---|
| Entity found, but **some of its records missed** | 0.0179 | **74 %** |
| Matched entity, nothing predicted | 0.0048 | 20 % |
| Singleton wrongly given records | 0.0015 | 6 % |
| Predicted, but nothing correct | 0.0001 | 0 % |

Among the entities that are only partly right, **336,476 lost recall only** (every record we gave was correct) versus 34,969 that added a
wrong record. So the leak is *weak records on entities we already found*, not false merges.

### Idea
A record with weak evidence (`p1 = 0.4`) is usually rejected on its own. But if its best entity already owns several strong records, it is far more
likely to be genuine than the same weak record on an entity nobody else points at. Neither pair-level scores nor a global threshold can see that.
So a small LightGBM model re-scores every query's best assignment using **entity-context features**, computed only from the scored table (no labels):

| Feature | Meaning |
|---|---|
| `p1`, `p2`, `gap` | the query's own evidence and competition |
| `n_strong`, `n_strong_s2`, `n_strong_s3` | how many *other* queries point at the entity with `p1 ≥ 0.9` (total, per vendor) |
| `n_mid` | ... with `p1 ≥ 0.5` |
| `max_other`, `sum_other` | strongest / total probability of the entity's other queries |
| `leak` | probability other queries put on this entity as their *runner-up* |
| `same_src_strong` | strong records of the same vendor (vendors repeat their own noise style) |
| `src` | S2 or S3 |

The output `r` is P(the query truly belongs to its best entity | its own evidence *and* the entity's context). The decoder is then just `r ≥ tau` (`tau = 0.7`,
tuned on half A). Singleton protection now emerges from the context: a lone weak record on an entity with no strong records gets a low `r`.

## 7.4 Comparing the decoders honestly

`scripts/tune_decode.py` splits the entities of the **full train world** into two halves by hash:

* **Half A** is used for everything that is *fitted*: thresholds, calibrators, `mu0`.
* **Half B** is used for every *reported* number.

Both decoders are evaluated with the exact metric emulation (08), counting every query (including queries of other
entities wrongly assigned to a B entity).

### Results (entity-level macro F0.5, half-B held-out entities: 1,103,761 of US + India, all 10.3 M train queries scored)

| Decoder | F0.5 | Singletons | Entities with matches | Matched precision | Matched recall |
|---|---|---|---|---|---|
| A. Threshold (`p1 ≥ 0.7`, gap 0) | 0.97558 | 0.9725 | 0.9758 | 99.49 % | 94.40 % |
| B. Expected-F0.5 (`mu0 = 0.02`) | 0.97559 | 0.9493 | 0.9772 | 99.49 % | 94.58 % |
| **C. Context stage (`r ≥ 0.7`)** | **0.97649** | **0.9836** | 0.9761 | 99.42 % | 94.84 % |

**Reading the table honestly**

* **B tied A (+0.00001).** The expected-F0.5 decoder gained recall on matched entities (0.9758 → 0.9772) but lost the same amount on singletons
  (0.9725 → 0.9493): its calibrated probabilities cannot tell a genuine lone record from a near-duplicate decoy attached to a singleton. Its best `mu0`
  also sat at the edge of the search grid (0.02), so it may be slightly under-tuned. We kept the code and tests (the maths is verified against brute force)
  but do not use it.
* **C gained +0.00091 over A** on entities never used for fitting. Nearly all of the gain is singleton protection (0.9725 → 0.9836) with a small recall gain.
* The gain is modest because precision is already 99.4–99.5 %: the remaining headroom is in *recall* (see 08).

`scripts/tune_decode.py` (A and B) and `scripts/tune_context.py` (C) reproduce these numbers.

## 7.5 One-to-one is guaranteed
In all three decoders every query is proposed to exactly one entity (its best), so a query can never appear in two
lists. `make_submission.py` asserts this before writing.

## 7.6 Hand-off
The accepted `(query, entity)` pairs are assembled into `matching_results.tsv` (`pipeline.assemble`): one row per
S1 entity in the same order as `test_source1`, comma-separated IDs, empty string for empty lists — the same layout
as the ground-truth file.
