# Documentation index — start here

This folder explains **every part** of the project, in the order data flows through it. If you read the files
in numeric order you will understand the whole system without opening any code; if you are looking for one
topic, the table below tells you where it lives.

## The problem in one paragraph
Amazon has a clean list of businesses (**Source 1**, "S1"). Two other vendors (**Source 2**, **Source 3**)
send us messy records describing *some of the same* businesses, with no shared ID. All we have is a business
**name** and an **address** per record. For every S1 business we must list which S2/S3 records describe it —
and leave the list **empty** if none do. The score is a per-business **F0.5** (mistakes that merge two different
businesses cost twice as much as missing a true match). Details: [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md).

## The solution in one picture

```
 raw TSVs (S1, S2, S3, country)                                   ┌──────────────────────────────┐
        │                                                         │  TRAINING SIDE (labels used) │
        ▼                                                         │                              │
 ┌───────────────┐   02      cleaned name / address text           │  matched pairs ──► alias     │
 │ Normalization │ ─────────────────────────────────────────────► │  table (04)                  │
 └───────────────┘                                                │  labelled candidate pairs    │
        │                                                         │        ──► LightGBM matcher  │
        ▼                                                         │            (06)              │
 ┌───────────────┐   04      Indic-script names rewritten to      │  full-world scores ──►       │
 │ Cross-script  │ ─────────  the S1 (Latin) spelling             │   calibrators + decoder (07) │
 │ aliases       │                                                └──────────────┬───────────────┘
 └───────────────┘                                                               │ fitted objects
        │                                                                        │ (models, tables)
        ▼                                                                        ▼
 ┌───────────────┐   03      ~28 plausible S1 entities            ┌──────────────────────────────┐
 │   Blocking    │ ─────────  for each S2/S3 record               │  INFERENCE (test, no labels) │
 └───────────────┘                                                │  same steps 02→03→05→06,     │
        │                                                         │  then decoding 07 →          │
        ▼                                                         │  matching_results.tsv        │
 ┌───────────────┐   05      ~60 numbers describing how           └──────────────────────────────┘
 │ Pair features │ ─────────  two records agree / differ
 └───────────────┘
        │
        ▼
 ┌───────────────┐   06      probability that this pair is a true match
 │    Matcher    │ ─────────
 └───────────────┘
        │
        ▼
 ┌───────────────┐   07      who gets which records, and who gets nothing
 │   Decoding    │ ─────────
 └───────────────┘
```

## Reading order

| # | Document | Question it answers |
|---|----------|---------------------|
| – | [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md) | What exactly are we asked to do, and how are we scored? |
| 01 | [`01_DATA_AND_FINDINGS.md`](01_DATA_AND_FINDINGS.md) | What does the data look like, and what did we discover about how it was made? |
| 02 | [`02_NORMALIZATION.md`](02_NORMALIZATION.md) | How do we clean the text so that "the same" strings actually look the same? |
| 03 | [`03_BLOCKING.md`](03_BLOCKING.md) | How do we avoid comparing every record with every other record? |
| 04 | [`04_CROSS_SCRIPT_ALIASES.md`](04_CROSS_SCRIPT_ALIASES.md) | How do we match a Hindi/Tamil/Telugu name against an English one — without a dictionary? |
| 05 | [`05_PAIR_FEATURES.md`](05_PAIR_FEATURES.md) | How do we turn "two records" into numbers a model can learn from? |
| 06 | [`06_MATCHER.md`](06_MATCHER.md) | How is the model trained and why is it built this way? |
| 07 | [`07_DECODING.md`](07_DECODING.md) | How do probabilities become the final lists — including the *empty* lists? |
| 08 | [`08_EVALUATION.md`](08_EVALUATION.md) | How do we measure progress honestly, and what are the results? |
| 09 | [`09_FRANCE_AND_RISKS.md`](09_FRANCE_AND_RISKS.md) | What can go wrong on the hidden test set, and what protects us? |
| 10 | [`10_RUNNING_THE_PIPELINE.md`](10_RUNNING_THE_PIPELINE.md) | Exactly which commands reproduce everything? |
| 11 | [`11_EXPERIMENT_LOG.md`](11_EXPERIMENT_LOG.md) | What did we try, what worked, what did not, and why? |
| – | [`GLOSSARY.md`](GLOSSARY.md) | What does each term mean? |
| – | [`METHODOLOGY.md`](METHODOLOGY.md) | The short write-up required for the final submission archive. |
| – | [`PLAN.md`](PLAN.md) | The original design plan (kept for the record; the numbered docs are authoritative). |

## How the sub-problems of the problem statement map to our components

The organizers describe five sub-problems. This table is the "no gaps" checklist.

| Sub-problem (from the problem statement) | Where we solve it | Doc |
|---|---|---|
| Records are noisy and inconsistent (typos, abbreviations, missing parts, other scripts) | Normalization + learned aliases + fuzzy features | 02, 04, 05 |
| Comparing everything with everything is too expensive → **blocking**, favouring recall | Multi-key rare-token blocking, ≈28 candidates per record | 03 |
| A **matching model** scores each candidate pair and discards lookalikes | LightGBM pair matcher with "leftover token" features | 05, 06 |
| **Singletons** (S1 businesses with no match) must get an *empty* list | Entity-context second stage: a lone weak record on an entity nobody else supports gets a low probability and is dropped | 07 |
| **Precision counts twice** (F0.5): when in doubt, do not merge | Decision thresholds tuned on an emulation of the real per-entity metric over the *full* train world | 07, 08 |
| **Region-specific patterns** (India vs US; France unseen) | Country partitioning, native-script handling, no country-specific rules, France safeguards | 01, 04, 09 |
| Pure ML: **no external databases / APIs / lookups** | Everything (aliases included) is learned from the provided data | 04, 09 |
| Deliverables: `matching_results.tsv`, `candidate_pairs.tsv`, runnable pipeline, methodology | `scripts/`, `outputs/submission/`, `METHODOLOGY.md` | 10 |

## Code map (where each idea lives)

| Idea | Module | Script |
|---|---|---|
| Reading files, caching cleaned records | `src/entity_resolution/io.py` | `scripts/build_interim.py` |
| Text cleaning | `normalize.py` | – |
| Candidate generation | `blocking.py` | `scripts/eval_blocking.py` |
| Alias learning / cross-script rewrite | `aliases.py` | `scripts/fit_aliases.py` |
| Pair features | `features.py` | – |
| Training data construction | `dataset.py` | `scripts/train_model.py` |
| End-to-end scoring | `pipeline.py` | `scripts/run_pipeline.py` |
| Simple threshold decoding | `decode.py` | `scripts/scan_decode.py` |
| Calibration | `calibration.py` | – |
| Expected-F0.5 decoding (tied the baseline; not used) | `expected_f05.py` | `scripts/tune_decode.py` |
| Entity-context second stage (best decoder) | `context.py` | `scripts/tune_context.py` |
| Leaderboard metric emulation | `metrics.py` | – |
| Submission writer | `pipeline.py::assemble`, `write_submission` | `scripts/make_submission.py` |
| Version history of uploads | – | `scripts/snapshot_submission.py`, `SUBMISSION_LOG.md` |
