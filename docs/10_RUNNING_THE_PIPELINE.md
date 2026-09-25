# 10 — Running the pipeline (reproducing everything)

## 10.1 One-time setup

```bash
uv sync                       # creates .venv with Python 3.12 and all dependencies (see pyproject.toml)
uv run pytest -q              # every test should pass
```

Place the organizers' files under `data/raw/tsv/{train,test}/` as
`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`,
`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`. **All files are tab-separated.**

**Where derived files go.** Normalized records are cached in `data/interim/`. Fitted artifacts (aliases, model,
calibrators, scored pairs) are written to `$ER_WORK` (default: `data/interim/`). If that directory is not
writable set `export ER_WORK=/some/writable/dir` before running anything.

## 10.2 The steps, in order

| # | Command | What it does | Time (16 cores) |
|---|---------|--------------|-----------------|
| 1 | `uv run python scripts/build_interim.py` | Clean all six data files → `data/interim/*.parquet` (02) | ≈ 1.5 min |
| 2 | `uv run python scripts/fit_aliases.py` | Learn cross-script aliases from train pairs → `aliases.parquet` (04) | ≈ 15 s |
| 3 | `uv run python scripts/eval_blocking.py --country US --n 100000` | (Optional) measure blocking recall (03) | ≈ 30 s |
| 4 | `uv run python scripts/train_model.py --n 300000 --rounds 600` | Build labelled pairs and train the matcher (06) → `matcher.txt`, `matcher_features.json` | ≈ 8 min |
| 5 | `uv run python scripts/run_pipeline.py --split train --countries US India` | Score the **full train world** (needed to tune decoding on the real metric) (07/08) | ≈ 45 min |
| 6 | `uv run python scripts/tune_decode.py` | Compare threshold vs expected-F0.5 decoders on held-out entities; save calibrators and parameters | ≈ 10 min |
| 6b | `uv run python scripts/tune_context.py` | Fit and evaluate the **entity-context second stage** (07, Decoder C); saves `context_model.txt`, `context_params.json` | ≈ 10 min |
| 7 | `uv run python scripts/make_submission.py --decoder context` | Score the **test** set (≈ 10 M queries) and write `outputs/submission/matching_results.tsv`. Add `--reuse` to skip countries already scored (it resumes after an interruption) | ≈ 40 min (≈ 5 min with `--reuse`) |
| 8 | `uv run python scripts/snapshot_submission.py --label ... --params ...` | Freeze the file with its sha256 and git commit and append a row to `docs/SUBMISSION_LOG.md` — **run right before every upload** | seconds |

The fitted artifacts needed by steps 6–7 (matcher, alias table, calibrators, context model, parameters) are also committed under `artifacts/`
(copy them into `$ER_WORK` to skip retraining). Step 7 accepts the chosen decoder parameters (see the script's `--help`) and prints per-country diagnostics
(09) and integrity checks (one row per S1 entity, each query assigned at most once).

## 10.3 Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ER_WORK` | `data/interim` | Where fitted artifacts and scored tables are written/read |
| `ER_ALIAS_MODE` | `native` | Which aliases are applied (`none`, `all`, `native_addr`, `native`) — see 04 |

## 10.4 Determinism
Sampling uses fixed seeds; fold assignment is a hash of entity ids; LightGBM is seeded by default. Re-running a
step on the same inputs reproduces its outputs (multi-threaded floating-point summation can cause tiny
last-digit differences in model scores).

## 10.5 Troubleshooting

* **`Permission denied` when writing inside the project.** The project sits on an NTFS volume that at one point
  flagged folders read-only. Run `chmod u+w data/interim scripts src src/entity_resolution tests docs outputs`
  (or point `ER_WORK` elsewhere).
* **Out of memory.** Lower `--chunk` in `run_pipeline.py` (default 300,000 queries) or `--n` in `train_model.py`.
* **Long jobs die when the session ends.** Launch them detached (`setsid nohup ... &`) and follow the log with `tail -f logs/<name>.log`.
  Scored results are saved per country, so `--reuse` resumes cleanly.
* **Empty lists must be truly empty.** The writer (`pipeline.write_submission`) switches CSV quoting off; a default CSV writer would emit `""`.
* **Parquet vs TSV.** Always use the TSVs; the Parquet copy stores empty addresses as `"nan"` (01).
* **Nothing scores / empty candidates.** Check the interim files exist and that `aliases.parquet` and
  `matcher.txt` are in `$ER_WORK`.
