# 08 — Evaluation: measuring progress honestly

**Code:** `src/entity_resolution/metrics.py` · **Tests:** `tests/test_metrics.py`

## 8.1 The metric we emulate

The leaderboard uses **macro-averaged F0.5 over S1 entities**. Our emulation (`metrics.macro_f05`):

* For an entity with **no true match** (singleton): score `1` if we predicted nothing, `0` otherwise.
* For an entity with true matches: `P = |predicted ∩ true| / |predicted|`, `R = |predicted ∩ true| / |true|`,
  `F0.5 = 1.25·P·R / (0.25·P + R)`; and `0` if we predicted nothing or nothing correct.
* The final score is the **plain average over entities** — an entity with 1 record counts as much as one with 9.

> **Caveat.** The problem video defines the metric in words but not the exact per-entity computation for several
> true matches. This is the natural reading; the first leaderboard score will confirm or correct it. The unit
> tests pin our reading so any change is deliberate.

## 8.2 Three levels of evaluation (and why we use all three)

| Level | What is measured | Speed | Use |
|---|---|---|---|
| **Blocking recall** | Is the true entity among the candidates? | seconds | design blocking (03) |
| **Record-level F0.5** | Each sampled query's accepted assignment as one decision | minutes | fast health check while developing features/models (06, 11) |
| **Entity-level macro F0.5 on the full train world** | Exactly the leaderboard's computation | ≈ 45 min | choose the decoder and its parameters (07) — the number that predicts the leaderboard |

Why the third level is necessary: with a *sample* of queries most entities appear once, so per-entity precision is
meaningless; and shrinking the world (fewer entities *and* queries) removes the near-duplicate decoys that cause
false merges, giving over-optimistic numbers. So the final tuning scores **every** query and **every** entity of
US and India.

## 8.3 Leakage control

* Matcher folds are hashes of the *true entity* (or query id for distractors): an entity never straddles
  training/validation (06).
* The alias table is fitted only on entities outside validation fold 0 (04).
* Decoder tuning splits entities into halves A/B: fitted on A, reported on B (07).

## 8.4 Known optimism in our numbers (so you can discount correctly)

* Record-level numbers come from a *sample* of queries (with every sampled query counted, including those blocking never
  reached); only the entity-level number reproduces the leaderboard computation, so it is the one to trust.
* France is absent from every validation number (09).
* Label noise sets a ceiling below 100 % (01).

## 8.5 Results

### 8.5.1 Headline (entity-level, full train world, held-out entities)

| Decoder | F0.5 on held-out entities |
|---|---|
| Threshold baseline (`p1 ≥ 0.7`) | 0.97558 |
| Expected-F0.5 | 0.97559 |
| **Entity-context stage** | **0.97649** |

Per country (threshold baseline over all entities): US 0.97688, India 0.97382. Details of each decoder are in 07.

### 8.5.2 Where the entity-level loss sits (threshold baseline, all 2,206,821 US+India train entities)

Total loss = 1 − F0.5 = 0.02435. Decomposed by failure type:

| Failure type | Loss | Notes |
|---|---|---|
| Entity found but records missed (partly right) | 0.01793 | 336,476 entities missed records only; 34,969 added wrong records only; 6,348 both |
| Matched entity, nothing predicted | 0.00481 | 0.51 % of matched entities |
| Singleton wrongly given records | 0.00153 | 2.73 % of singletons |
| Predicted but zero correct | 0.00009 | |

Entities with a *single* true record score lowest (F0.5 0.933 vs 0.971 for 2 records, 0.978 for 3, ≈ 0.984 for 7+): a lone record is
either found or it is a total miss, so there is nothing to average out.

### 8.5.3 Where true matches get lost (record level, 7,638,365 true matches, `p1 ≥ 0.7` rule)

| Fate of a true match | Share |
|---|---|
| Accepted | 94.43 % |
| No plausible candidate at all (blocking miss or every score below 0.02) | 2.14 % |
| Right entity is best but `p1` in [0.02, 0.4) | 1.12 % |
| Right entity is best but `p1` in [0.4, 0.7) | 1.04 % |
| Wrong best entity (a look-alike outranks the true one) | 0.91 % |
| True entity is only the runner-up | 0.38 % |

Of the 163,084 records with no plausible candidate: 30 % have an empty address (largely irreducible), 15 % have a native-script name; the
remaining ≈ 55 % are noise-heavy records that blocking never surfaced. India contributes 88 k of them, the US 75 k. This is the largest
remaining lever, and is the reason for the pending blocking-cap experiment (11.5).

### 8.5.4 The test set is composed differently from train (and why acceptance is "only" 57 %)

| | Train (US+India) | Test (US+India) | Test (France) |
|---|---|---|---|
| S2/S3 records per S1 entity | 4.68 | **5.79** | 5.53 |
| Records with any plausible candidate | 76.8 % | 64.7 % | 70.5 % |
| Records we accept | 70.3 % | 56.6 % | 59.0 % |
| Entities with an empty list | 5.91 % | 5.97 % | 5.67 % |
| Records per non-empty entity | 3.50 | 3.49 | 3.46 |

Acceptance is a *consequence* of composition, not a goal: it equals (true matches × recall) / (all records). If test entities have the
train structure (5.58 % singletons, 3.67 true matches per matched entity), the test set holds `1,732,544 × 0.9442 × 3.67 = 6.00 M` true
matches (60 % of its records, vs 74 % in train), and at our 94.4 % recall we expect 5.667 M accepted records; we accepted 5.674 M. The
per-entity histogram of assigned records is also almost identical between train and test (e.g. 3 records: 24.4 % vs 24.4 %; 0 records:
5.9 % vs 6.0 %). Both checks are consistent with "test simply has ≈ 40 % non-matching records"; neither can *prove* it without labels.

### 8.5.5 Record-level history (fast development metric; see 11 for the full log)
0.9747 (baseline) → 0.9787 (leftover features) → 0.9820 (native-script aliases) → 0.9827 (300 k training queries per country).
