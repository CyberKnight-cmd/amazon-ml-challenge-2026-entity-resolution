# 09 — France, other risks, and rules compliance

This document lists what could go wrong on the hidden test set and what we did about each item. Where we
*cannot* test something offline, we say so plainly.

## 9.1 The France problem

The test set contains **France** (15 % of S1 ≈ 259,000 entities, ≈ 700,000 S2/S3 records) which **never appears
in training** (train = US 60 %, India 40 %). Anything the model learned about "how noise looks" was learned from
US and Indian data. Three things could break:

1. **Different conventions.** French addresses look like `20 Rue de Chateaudun, Tourcoing, Hauts-de-France`;
   abbreviations differ (`R.` for `Rue`), accents are common (`Collège`), articles (`de`, `la`) are frequent.
2. **Different noise mix.** The data generator may corrupt French records differently.
3. **Calibration shift.** Even if ranking still works, a model probability of 0.9 might mean 0.8 in France, so a
   threshold tuned on US/India would let too many wrong merges through.

### What protects us (built)
* **No country input anywhere.** No feature, key or rule reads the country name, except that records are
  *partitioned* by country (matches never cross countries), which only shrinks the problem.
* **No hand-written language rules.** Normalization (02) only lowercases, folds Latin accents, and removes
  punctuation — all valid for French. `Collège` → `college` on both sides.
* **Language-agnostic features.** Character-level similarities, token overlaps, numbers, and "leftover" analysis
  do not care what language the words are.
* **Blocking is frequency-based.** French filler words (`rue`, `de`, `la`) are automatically down-weighted by IDF
  because they occur in many records, exactly like `street` or `plot no` in the other countries.
* **Diagnostics in `make_submission.py`.** For every country it prints queries accepted, acceptance rate and the
  share of empty lists. If France's acceptance rate is far from US/India's, calibration has shifted and we
  will see it immediately.
* **A stricter France threshold** (`--france_t`) is available for the threshold decoder, and the expected-F0.5
  decoder (07) can be fed a more conservative prior for unseen countries.

### What we cannot verify offline
There are **no French labels**, so France's true accuracy is unknowable before the leaderboard reports it. Our
mitigation is conservative decisions plus the diagnostics above.

### Contingency if France looks off (not yet needed / not yet run)
*Transductive self-training:* take France test queries the model is extremely sure about (p ≥ 0.99, large gap
over the runner-up), treat them as pseudo-labels, and (a) check whether their statistics look like US/India
matches, (b) optionally refit calibration on them. This uses only the provided test *inputs* (no labels, no
external data), which the rules allow.

## 9.2 Other risks

| # | Risk | Consequence | Status / mitigation |
|---|------|-------------|---------------------|
| 1 | **Data provenance.** Our copy came from a third-party upload, not the official portal | Could differ from what the leaderboard uses | Structure matches every statement in the problem video; the organizers' validator and first leaderboard score are the real check. Re-download from the portal if anything mismatches |
| 2 | **Validation script not yet obtained** | A formatting error could waste a submission | We check ourselves (one row per S1, IDs exist, each query at most once, header as in the ground-truth file); the official script must still be run |
| 3 | **Scorer definition for entities with several true matches is not published** | Our thresholds could be tuned to a slightly different metric | We emulate the natural reading (per-entity P, R, F0.5, macro-averaged, singleton rule) and unit-test it (`tests/test_metrics.py`); the first leaderboard score confirms or corrects it |
| 4 | **`candidate_pairs.tsv` format unspecified** | Audit file could be rejected | We assume `source1_entity_id<TAB>candidate_entity_id`; adjust when the official format is known |
| 5 | **Label noise** (some "matches" only share an address) | Caps achievable recall | Accepted; we do not chase it |
| 6 | **Density mismatch between train and test worlds** | Model calibrated on a differently crowded world | Training uses the *full* S1 index per country, so neighbour density is realistic; entity-level tuning runs on the *full* train world, not a shrunken one |
| 7 | **The test set has more non-matching records than train** (5.8 S2/S3 records per entity vs 4.7; ≈ 40 % vs 26 % unmatched, inferred) | Decoy density and the prior of a candidate being genuine are higher than in the data our thresholds were tuned on | Per-entity behaviour on test matches train (empty lists 6.0 % vs 5.9 %; records per entity 3.49 vs 3.50) and the extra records mostly lack any plausible candidate (64.7 % vs 76.8 % have one), i.e. they are easy to reject (08, 8.5.4). Unverifiable without labels; the first leaderboard score is the check |
| 8 | **Filesystem quirk** (NTFS volume marked new directories read-only mid-session) | Scripts fail with `Permission denied` | Documented in `10`; fix is `chmod u+w` on the project directories |

## 9.3 Compliance with "pure ML, no external data"

| Question | Answer |
|---|---|
| Any external database, API, dictionary or lookup? | **No.** All aliases (including native-script ones) are learned from the provided training pairs. A hand-written abbreviation table and a transliteration library were deliberately *not* used (removed `unidecode`). |
| Any pretrained model or embeddings? | **No.** |
| Which third-party code is used? | Open-source *libraries* only: `polars` (data), `rapidfuzz` (string distance algorithms), `lightgbm` (learning), `numpy`/`scipy`/`scikit-learn`, `pyarrow`. None carries data about businesses or languages. |
| Are test labels used? | No (there are none). Test *inputs* are used only for inference. |
| Copying others' code | Other teams' public repositories exist online. **None of their code or approaches was used.** (While searching for the official dataset link we opened one repository page; it contained no code we used.) |
