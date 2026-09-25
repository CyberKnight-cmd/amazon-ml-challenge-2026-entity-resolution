> **Status note.** This is the *original* plan, kept for the record. What was actually built (and measured) is described by the numbered
> documents `01`-`11`; where they disagree, they are authoritative. Notably: aliases are applied only for native-script tokens (04), and an
> entity-context second stage (07) replaced the planned expected-F0.5 decoder as the best decoder.

# Solution plan

## 1. Formulation
Ground truth is 1-to-1 from the noisy side (every S2/S3 record matches <= 1 S1 entity) and never crosses
countries. So we do **not** compare S1 against S2/S3 entity by entity. We treat each S2/S3 record as a query:

> retrieve candidate S1 entities in the same country, score them, take the best one **or none**.

Each S1 entity's answer is then the union of the S2/S3 records that chose it. This gives the 1-to-1 constraint
for free, keeps candidate sets small, and makes "no match" (distractors and singletons) a first-class outcome.

The metric is precision-tilted (F0.5) and a singleton is all-or-nothing, so every decision rule is tuned on
an emulation of the metric, not on pair-level accuracy.

## 2. Pipeline
| # | Stage | Method | Gate |
|---|---|---|---|
| 0 | Normalize | fold accents, punctuation, drop N/A components, collapse repeated tokens, domain-style names -> one token. No country rules, no lookup tables (France is unseen). | done, tests pass |
| 1 | Blocking | per-country multi-key joins on **rare** tokens (address and name), squashed-name prefix, house-number + rare street token. Score candidates by IDF-weighted shared keys, keep top-K per query. Polars joins, chunked by country. | recall of true pairs >= 99%, mean candidates per record <= ~30 |
| 2 | Pair features | rapidfuzz `cpdist` (multithreaded C++): ratio / token-set / partial / Jaro-Winkler on name, squashed name and address; IDF-weighted token overlap; house-number equality; missing-address flags; length ratios; block-key counts. | throughput >= 1M pairs/s |
| 3 | Matcher | LightGBM binary classifier, split **grouped by S1 id**. Second pass adds within-query context: score rank, gap to runner-up, number of strong candidates. | pair-level precision/recall at the chosen operating point |
| 4 | Decoding | per record: accept best candidate iff p >= t1 and gap to runner-up >= g; per entity: singleton guard (drop everything if the best accepted record is weak). t1, g and guard tuned on emulated macro F0.5. | validation F0.5 |
| 5 | Outputs | `matching_results.tsv` (one row per S1, empty list allowed), `candidate_pairs.tsv`, validator. | organizers' validator passes |

## 3. Known hard parts and how we handle them
- **Address noise** (only ~9% exact match; house numbers corrupted ~25%): token-level and fuzzy address
  features, never exact-address keys alone. House-number match is a feature, not a gate.
- **Native-script India records** (~24% of India S2, ~13% of S3 names; S1 always Latin): most still match through
  the Latin address. For the rest, learn a native->Latin token alignment **from training pairs** (counts, not a
  library) and measure how much recall it recovers before keeping it.
- **Domain-style names** (`kimb1eolvas.com`): squashed-name comparison and prefix keys.
- **Learned aliases** (st/street, tx/texas, pvt/private, native state names): mined from the token differences of
  training pairs, canonicalized towards the S1 spelling (S1 is the clean source).
- **France domain shift**: no country feature, no hard-coded US/India rules, language-agnostic string
  features. Monitor acceptance rate and score distribution per country on test; mine France aliases
  transductively from very-high-confidence test pairs (labels not needed). Use a stricter threshold for France
  if its scores look inflated.
- **Label noise** (some "matches" share only the address): cap the recall we chase; do not tune to noise.

## 4. Compute budget
- Whole pipeline CPU-only; polars for joins, rapidfuzz for scoring, LightGBM for the model; no GPU, no embeddings.
- Work per country in chunks; parquet intermediates.
- Iterate on a 10% "dev world" (10% of S1 entities, their matched records, and 10% of distractors); report final
  numbers on the full train set and grouped cross-validation.

## 5. Evaluation
- `src/entity_resolution/metrics.py`: emulate the leaderboard (per-S1 P/R/F0.5, macro average, singleton rule).
- Blocking is judged on recall of true pairs and candidates per record, and the loss is broken down by country
  and by noise type.
- Every model change is compared on the same grouped split.

## 6. Risks
1. Data provenance unverified; the first leaderboard score is the real check.
2. The scorer's exact per-entity P/R definition is not published (see PROBLEM_STATEMENT.md, section 10).
3. `candidate_pairs.tsv` format is not specified; we assume `source1_entity_id<TAB>candidate_entity_id`.
4. France cannot be validated offline; the safeguard is conservative thresholds plus per-country monitoring.
