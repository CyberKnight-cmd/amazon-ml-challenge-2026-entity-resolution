# Methodology — Business Entity Resolution (Amazon ML Challenge 2026)

_Required 1–2 page write-up. Every claim below is expanded, with data and code pointers, in the numbered documents `01`–`11`._

## 1. Approach in one paragraph
We treat the task as **per-query assignment**: every S2/S3 record ("query") may belong to at most one S1 entity and never crosses countries, so
for each query we retrieve a short list of plausible entities (blocking), score each (query, entity) pair with a gradient-boosted model, and then
decide — per entity — which records to list, including the *empty* list for singletons. All components are learned from the provided data only;
no external database, API, dictionary, transliteration table or pretrained model is used.

## 2. Pipeline and models
| Stage | Method | Doc |
|---|---|---|
| Normalization | lowercase, fold Latin accents, punctuation removal, drop `N/A`/`NULL` components, collapse repeated tokens, strip zero-padding of numbers, detect website-style names. No country-specific rules (France is unseen in training). | 02 |
| Cross-script aliases | Native-script (Indic) name tokens are rewritten to the S1 spelling using an alias table **learned from training pairs** (align leftover tokens in order, count, keep dominant partners: 3,096 name / 3,415 address aliases; only the native-script ones are applied). | 04 |
| **Candidate generation (blocking)** | Per country. Seven key types (name tokens, address tokens, adjacent-name bigrams, squashed-name prefix, rare address-token pairs, number×word pairs, order-insensitive whole-address). Keys shared by too many S1 entities are dropped; candidates ranked by IDF-weighted key evidence, top 30 per query. **Recall 98.4 % (US) / 96.4 % (India), ≈ 28 candidates per query.** | 03 |
| **Pair features (≈ 60)** | Blocking evidence; name/address similarities (Levenshtein-ratio family, token sort/set, Jaro-Winkler, squashed-name partial ratio); number features; **leftover-token features** (similarity of what remains after removing shared words/numbers — separates typos from substituted words); S1-side ambiguity counts; record-quality and native-script flags. No feature reads the country. | 05 |
| **Matcher** | LightGBM (255 leaves, 421 rounds), trained on 17 M labelled candidate pairs from a random sample of 300 k queries per country against the *full* S1 index; folds grouped by true entity. | 06 |
| **Decoding** | Entity-context second stage: a small LightGBM re-scores each query's best assignment using entity-level context (number of strong records the entity already has, per vendor; strongest other record; probability leaked as others' runner-up); accept if `r ≥ 0.7`. Singleton protection emerges from the context. | 07 |

## 3. Experiments (selected; full log in `11`)
| Change | Metric | Result |
|---|---|---|
| Blocking: per-key-type caps, number×word keys, order-insensitive address key | recall US / India | 97.8 → 98.4 % / 94.2 → 96.0 % |
| + leftover-token features | record-level F0.5 | 0.9747 → 0.9787 |
| Alias modes on identical queries: none / all / native+address / **native only** | record-level F0.5 | 0.9777 / 0.9782 / 0.9809 / **0.9820** |
| 150 k → 300 k training queries per country | record-level F0.5 | 0.9820 → 0.9827 |
| Decoders on the **full US+India train world**, held-out entities (1.10 M) | entity-level macro F0.5 | threshold 0.97558; expected-F0.5 0.97559 (tie); **context stage 0.97649** |

Negative results kept in the record: rewriting Latin abbreviations (`texas → tx`) slightly *hurt* the US; an expected-F0.5 decoder tied the threshold.

## 4. Where the remaining error is (measured)
74 % of the entity-level loss is *records missed inside entities we already found* (not false merges); precision is already ≈ 99.4 %. Of all true
matches, 94.4 % are accepted, 2.1 % never reach a plausible candidate (30 % of those have an empty address), ≈ 2.2 % score low, ≈ 0.9 % lose to a
look-alike. See `08`.

## 5. Handling the region shift (France)
France (15 % of test S1) never appears in training. We use no country input or hard-coded language rules, only frequency-based blocking and
language-agnostic features; on test, France's acceptance rate (60.0 %), empty-list share (6.0 %) and confidence profile match the trained
countries. True France accuracy cannot be verified offline (09).

## 6. Conclusion and limitations
Blocking plus a leftover-aware gradient-boosted matcher and an entity-context decoder reach ≈ **0.976 entity-level F0.5** on held-out training
entities using only CPU and the provided data (≈ 40 min to score 10 M test records on 16 cores). Limitations: the scorer's exact per-entity definition is
unpublished (we emulate the natural reading, unit-tested); the test set has proportionally more non-matching records than train (≈ 40 % vs 26 %,
inferred); ≈ 2 % of true matches are unreachable by blocking (looser caps not yet evaluated); France is unvalidated.

## 7. Reproducibility
`uv sync`, then follow `docs/10_RUNNING_THE_PIPELINE.md`. Fitted artifacts are under `artifacts/`; every uploaded file is recorded with its sha256 and
git commit in `docs/SUBMISSION_LOG.md`. Source functions are documented by docstrings; 22 unit tests cover normalization, metric emulation,
alias learning, decoding, the context features and the exact output file format.
