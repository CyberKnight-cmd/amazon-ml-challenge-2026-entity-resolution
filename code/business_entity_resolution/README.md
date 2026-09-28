# Amazon ML Challenge 2026 — Business Entity Resolution

Link every business in a clean reference list (**Source 1**) to its matching records in two noisy vendor feeds
(**Source 2**, **Source 3**) using only names and addresses — and leave the list *empty* when nothing matches.
Scored by per-entity **macro F0.5** (precision counts about twice as much as recall). No external data, APIs or lookups.

**Result so far:** ≈ **0.9765 entity-level F0.5** on held-out *training* entities (full US+India train world, 1.1 M entities). Test-set accuracy is
unknown until the leaderboard reports it; France is unseen in training.

## How it works (one minute)
```
clean text → learn cross-script aliases → block (≈28 candidates per record) → pair features → LightGBM matcher → entity-context decoder → matching_results.tsv
```
Everything is CPU-only: ≈ 40 minutes to score the 10 M test records on 16 cores.

## Where to start
* **New here?** Read [`docs/README.md`](docs/README.md) — a guided tour in reading order, with a sub-problem-to-component map.
* **Just want to run it?** [`docs/10_RUNNING_THE_PIPELINE.md`](docs/10_RUNNING_THE_PIPELINE.md).
* **The 1–2 page approach write-up:** [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).
* **What we tried and measured:** [`docs/11_EXPERIMENT_LOG.md`](docs/11_EXPERIMENT_LOG.md).

## Layout
```
docs/                  numbered documentation 01-11, PROBLEM_STATEMENT, METHODOLOGY, GLOSSARY, SUBMISSION_LOG
src/entity_resolution/ the pipeline package (normalize, blocking, aliases, features, dataset, pipeline,
                       decode, calibration, expected_f05, context, metrics)
scripts/               entry points (build, fit, train, tune, score, snapshot) - see docs/10
tests/                 22 unit tests
artifacts/             fitted matcher, alias table, calibrators, context model (small, committed)
data/                  raw / interim / processed (git-ignored; the dataset is never committed)
outputs/               submission files (git-ignored) - every upload is logged in docs/SUBMISSION_LOG.md
```

## Setup
```
uv sync
uv run pytest -q
```
All data files are **tab-separated**; place them under `data/raw/tsv/{train,test}/` (see docs/10).
