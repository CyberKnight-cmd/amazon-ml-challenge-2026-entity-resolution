# 06 — The matcher: a model that scores each candidate pair

**Code:** `src/entity_resolution/dataset.py`, `scripts/train_model.py`, `pipeline.py` · **Model:** LightGBM
(gradient-boosted decision trees)

## 6.1 The problem this solves

Blocking hands us ≈ 28 candidate entities per query, of which at most one is correct (and often none, when the
query is a distractor). The matcher answers, for each pair independently:

> *"How likely is it that this query is a noisy copy of this entity?"*  → a probability `p` in [0, 1].

## 6.2 Why gradient-boosted trees (and not something fancier)

* The inputs are ≈ 60 heterogeneous numbers (similarities, counts, flags, missing values) — the sweet spot of
  gradient-boosted trees, which routinely beat neural networks on tabular data.
* Handles missing values natively (5.3), needs no scaling, trains in minutes on CPU, predicts millions of rows per
  minute, and its behaviour is inspectable (feature importances).
* Fits the "least computation" goal: no GPU, no embeddings, no pretrained models (which would also risk the
  "no external resources" rule).

## 6.3 How the training data is built (`dataset.py`)

We mimic the real situation as faithfully as possible:

1. Pick a country. Take the **complete S1 index** of that country (so the number of look-alike neighbours is
   realistic).
2. **Sample queries** (S2/S3 records) at random — 300,000 per country for the final model.
3. Run the *same* alias rewriting, blocking and feature code that inference uses. (Training and inference share
   code, so there is no train/serve mismatch.)
4. **Label** each candidate pair: `1` if the entity is the query's true owner according to the ground truth, else
   `0`. Distractor queries only produce `0`s.

Result ≈ 17 million labelled pairs, ≈ 2.5 % positive. (Roughly ≥ 97 % of true matches are among the candidates —
that is the blocking recall from 03.)

**Fold assignment (leak-proofing).** Every pair gets a fold `0–4` from a hash of its *true entity* (or of the
query id for distractors). All pairs of an entity therefore fall in the same fold. We train on folds 1–4 and
validate on fold 0, so the model never sees an entity — or its near-duplicate distractors — in both training and
validation.

## 6.4 Training

* Objective: binary log-loss. 255 leaves, learning rate 0.08, feature and row subsampling 0.8,
  `min_data_in_leaf` 200, L2 = 1, early stopping on the validation fold (stops ≈ 420 rounds).
* Trains in ≈ 3 minutes on 16 cores.
* Saved as `matcher.txt` plus `matcher_features.json` (the exact feature order — the model is useless without it).

Validation log-loss fell as we improved the system: 0.0048 (first version) → 0.0040 (leftover features) →
0.0029 (more training data + cross-script aliases; not strictly comparable because the validation sample changed).

## 6.5 What the model relies on (feature importance by gain)
1. `score` — blocking evidence weighted by rarity (by far the strongest);
2. `sq_partial`, `max_idf`, `n_tsort`, `a_tset`, `n_keys`;
3. the leftover/number features: `a_rest_len2`, `m_rest_len1/2`, `num_first_eq`, `num_ratio`.

The leftover and number features rank in the top dozen even though the simple similarities were already present —
confirming they carry information the classic features lacked.

## 6.6 Inference (`pipeline.score_country`)

For each country: build the S1 index once; stream the country's queries through in chunks of 300,000:
`prep_queries` (flags + aliases) → `query_candidates` (blocking) → `pair_features` → `model.predict`. We keep
**every pair with p ≥ 0.02** (≈ 1.1 pairs per query) — not just each query's best — because the decoders in 07
need the runner-up and how much probability sits on *other* entities. Cost: ≈ 78 s per 300,000 queries.

## 6.7 What the matcher does **not** do
It scores each pair alone. It does not know that a query can have only one owner, nor that an entity is scored as a
whole. Those are decoding problems, solved in 07.

## 6.8 Hand-off
Scored pairs `(query, entity, p)` go to decoding (07).
