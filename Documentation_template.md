# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** 4seekers
**Team Members:** Arya Gupta, Juhi Jaiswal, Avirupa Malakar, Shlok Gupta  
**Submission Date:** 27 September 2026  
**Repository:** https://github.com/CyberKnight-cmd/amazon-ml-challenge-2026-entity-resolution

---

## 1. Executive Summary

A CPU pipeline links every Source-2/3 record to at most one Source-1 entity in four steps:

1. **Generator-aware blocking.** Two blockers run side by side: IDF-weighted token keys, plus keys built to invert the dataset's corruptions.
2. **Stage-1 matcher.** A LightGBM pair classifier scores every candidate.
3. **Stage-2 re-ranker.** A second LightGBM model re-scores the plausible pairs with costlier evidence. That includes a fine-tuned multilingual cross-encoder (`intfloat/multilingual-e5-small`, MIT licence, 118 M parameters).
4. **Decision.** Each record goes to its best entity only if the re-ranker is confident, which favours precision as F0.5 requires.

The core innovation is blocking keys designed around how the noisy records are generated, backed by an analysis of why the leaderboard and offline validation disagreed.

## 2. Methodology

### 2.1 Problem Analysis
- Every S2/S3 record belongs to **at most one** S1 entity, and matches never cross countries. We therefore decide per record and partition by country.
- About 25% of train records are **decoys**: near-copies of real entities with one deliberate change, such as a swapped name word or a different house number.
- Name noise includes case and accents, digit-for-letter typos (`5afe`), legal-suffix swaps, reordered and duplicated words, glued or domain-style names, "X aka/DBA Y" prefixes, and full transliteration into Indic scripts.
- Address noise includes abbreviations, state renaming, reordering, truncation and house-number edits.
- **Each source's copies of an entity share one perturbed variant** (e.g. an S3 house number `4210` where S1 has `4211`). A genuine record therefore repeats rare divergences that its siblings also carry, while a decoy does not.
- **The test set is structured differently from train:**
  - Test S1 is about half the size per country (US 0.66 M vs 1.32 M).
  - Test has 5.8 records per entity against 4.7.
  - 35% of test records have no plausible match against 23%.
  - France appears only in test.

### 2.2 Solution Strategy
**Approach Type:** Blocking, then a pairwise classifier, then a re-ranker, then a precision-weighted decision (hybrid).

**Core Innovation:** Blocking keys that invert the generator's corruptions. They recover 53–62% of the true matches that generic keys miss.

## 3. Candidate Generation (Blocking)
- **v1 keys:** name and address tokens, adjacent name-word pairs, squashed-name prefix, pairs of rare address tokens, number × word, and the sorted address. A key is dropped when shared by more S1 entities than a per-type cap. The top 30 candidates by summed IDF are kept.
- **v2 keys (generator-aware):**
  - name bags: the distinct name tokens with up to two deletions per side;
  - anagrams of those bags, for glued names;
  - fuzzy house numbers (one digit deleted) × rare address word;
  - rare name token × rare address word;
  - squashed-name prefix × rare address word;
  - single-typo token variants.

  Before keys are built, digits inside words are read as letters, and learned alias classes (ltd/limited, st/street/saint, texas/tx) are applied to both sides. The v2 top 15 are added to the v1 top 30.
- **Candidates generated:** about 38–40 per record at blocking. The candidate file lists the pairs the final model scores (stage-1 `p ≥ 0.02`): 7.5 M pairs, about 4.4 per S1 entity.
- **How true matches were protected:** measured on 100k sampled train records per country, recall is **US 98.44% → 99.41%** and **India 96.83% → 98.52%** with v2. Most of what remains unreachable is empty-address records with a common name, which no method can place.

## 4. Matching Model
**Features used:**
- **Name features:** rapidfuzz ratio, token-sort, token-set, Jaro-Winkler, squashed-name ratio and partial ratio. Also "leftover" tokens: what remains after removing shared words (a typo leaves two similar strings, a swapped word two dissimilar ones), and how common those leftover words are among S1 names.
- **Address features:** the same similarities, plus house-number relations (difference, prefix, suffix), shared numbers, and leftover-number distance.
- **Other:**
  - blocking evidence (v1 and v2 key counts and IDF sums);
  - noisy-channel log-likelihood ratios of the edits between the two records, learned from training pairs;
  - sibling agreement (whether the record repeats rare divergences shared by the entity's confident records);
  - the cross-encoder probability.

**Model type:**
- **Stage 1:** LightGBM binary classifier.
- **Stage 2:** LightGBM re-ranker.
- **Cross-encoder:** `multilingual-e5-small` fine-tuned for 3 epochs on 310 k uncertain pairs (stage-1 `p ≤ 0.98`). On held-out uncertain pairs it reaches AUC 0.9555 against 0.9160 for stage 1, and 0.970 combined.

**Threshold selection method:** entity-level macro F0.5 on held-out entities (entities split by hash into a fitting half and a scoring half). A record is accepted only if its best entity's re-ranker score passes the threshold. We used a stricter threshold (0.85) than the offline optimum, because the test set holds more unmatched look-alikes.

## 5. Results & Error Analysis
- **F0.5 (macro), held-out entities of the train world:**

  | Configuration | Held-out F0.5 |
  |---|---|
  | Stage-1 threshold | 0.97558 |
  | Stage-1 threshold + v2 blocking | 0.98061 |
  | Stage 2 + v2 blocking | 0.98882 |
  | + cross-encoder | 0.98934 |
  | Stage 2 without entity-context features (final) | 0.98865 |

- **Public leaderboard:** 0.9830 for the final file.
- **Why offline and public disagree:** the v2 key evidence sums log(S1 size ÷ key frequency) over many shared keys, and its frequency caps are absolute. On the test set's smaller S1 (US halves, France is 259 k) the evidence is deflated and the stage-1 probability under-confident. We measured this: the share of records whose best candidate reaches `p ≥ 0.7` falls by 0.095 in the US for v2 against 0.061 for v1, while India, whose S1 is about the same size, is unaffected. Stage 2 repairs most of it.
- **Common false positives:** decoys with one swapped name word, or a changed house number at an otherwise identical address.
- **Common false negatives:**
  - empty-address records whose name is shared by several S1 entities (about 1.5% of true pairs; unresolvable);
  - names transliterated into Indic scripts with truncated addresses;
  - heavily misspelled names.

## 6. Scalability to New Countries

Amazon runs entity resolution across many markets, not just the US, India and France. Our pipeline is built to add a country by adding data, not by adding code.

**The architecture is already country- and language-agnostic.**
- **No country feature, key or rule anywhere.** Country is used only to *partition* the problem (matches never cross countries), which shrinks each sub-problem rather than complicating it. France — never seen in training — is scored by the same model with zero France-specific logic.
- **No hand-written language rules.** Normalization only lowercases, folds Unicode accents and strips punctuation — rule-free by design. It currently preserves Latin and Indic scripts; supporting another script (Arabic, Cyrillic, CJK, Thai) is a one-line widening of a single Unicode character range in `normalize.py`, not new logic. Every matching feature is character-level similarity, token overlap, number relations, or "leftover" analysis — none of which reads the language.
- **Blocking adapts to a new language automatically.** Keys are weighted by IDF, so a market's filler words (`rue`/`de` in French, `plot no` in India, `straße` in Germany) are down-weighted purely from their frequency, with no per-country stop-word list.

**Scaling out is embarrassingly parallel.** Because everything partitions by country, N markets are N independent shards. The pipeline is CPU-only and blocking is sub-quadratic (inverted-index keys with frequency caps, ~40 candidates/record), so cost grows roughly linearly in total records and shards run concurrently on commodity hardware. Adding a market adds a shard; it does not slow the existing ones.

**The one data-dependent component generalizes by re-fitting, not re-coding.** The learned alias table (cross-script equivalences such as `लिमिटेड → limited`) is *counted* from that country's own labelled pairs by `fit_aliases.py`; the alignment algorithm is language-agnostic, so a new market's aliases come from re-running the same script on its training pairs — no dictionary, no external API. The only code touch for a script outside Latin/Indic is the one-line Unicode-range widening in `normalize.py` noted above; the alias learning and every downstream feature are unchanged. Latin-script markets need neither aliases nor a normalization change — France, unseen in training, already runs through the pipeline unmodified. The cross-encoder base model (`multilingual-e5-small`) is multilingual (~100 languages), so a new language is representable without architecture changes; per-market fine-tuning further improves it when labels exist.

**Calibration is the real cross-market risk, and we have a concrete plan for it.** §5 showed that features scaling with S1 index size are deflated on a smaller index, making probabilities under-confident. The lesson generalizes: **any feature that scales with country size must be normalized before it transfers**, and thresholds must be calibrated per country rather than shared globally. For a *cold-start country with no labels* (France's situation) the recipe is: (1) a stricter, more precision-favouring threshold by default; (2) per-country diagnostics — acceptance rate and empty-list share — that flag calibration drift the moment a new market looks unlike the calibrated ones; and (3) *transductive self-training*: take the new market's highest-confidence test predictions (large margin over the runner-up), treat them as pseudo-labels, and refit calibration and, if worthwhile, the alias table — using only provided test *inputs*, never external data.

**Onboarding checklist for a new country.** Point the pipeline at that country's S1 index and S2/S3 records → run `fit_aliases.py` if the script is non-Latin → score with the existing models → verify per-country diagnostics against known markets → set the per-country threshold (strict until labels or leaderboard feedback arrive). No feature, model or normalization rule is rewritten.

## 7. Conclusion
Blocking built around the data generator raised recall substantially. A cross-encoder added the strongest per-pair evidence. The main lesson is that validation must reproduce the test set's structure: its S1 size, the rate of unmatched records, and records per entity. Features that scale with S1 size must be normalized before they can transfer.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:

| Path | Contents |
|---|---|
| `src/entity_resolution/` | `normalize`, `aliases`, `blocking` (v1), `blocking_v2`, `features`, `pipeline`, `rerank` (stage 2 and sibling features), `channel` (noisy-channel features), `cross_encoder`, `metrics` |
| `scripts/` | Entry points, in order: `build_interim.py`, `fit_aliases.py`, `train_model.py`, `run_pipeline.py` (train and test), `stage2_build.py`, `ce_train.py`, `ce_score.py`, `stage2_train.py`, `stage2_apply.py`, and `validate_submission.py` (the organizers' validator) |
| `artifacts/` | The fitted alias table, the stage-1 matcher, the stage-2 model with its features and channel table, and the fine-tuned cross-encoder |

`README.md` gives the exact commands, and `requirements.txt` pins every package version.

### B. Additional Results
`docs/03_BLOCKING.md` covers the blocking design and recall, and `docs/11_EXPERIMENT_LOG.md` every measured experiment, including negative results. `docs/SUBMISSION_LOG.md` records each leaderboard upload with its file sha256 and code commit.
