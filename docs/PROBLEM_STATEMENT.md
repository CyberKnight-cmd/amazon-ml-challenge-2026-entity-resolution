# Amazon ML Challenge 2026 — Business Entity Resolution

_Source: the organizers' problem-statement walkthrough video (transcribed). Section 9 holds facts we
measured ourselves on the data; everything before it is from the organizers._

## 1. Context
A business signs up on Amazon Business; we capture its name and address. To enrich it, records are pulled
from other data vendors, each with its own formats and conventions. The external sources share **no common
identifier** with ours, so the only usable fields are **business name** and **business address** (the data is
deliberately limited to these two).

The same business is written differently by each source (e.g. "Acme Robotics Incorporated", an abbreviated
address, a nearby landmark instead of a street).

## 2. Sources
| Source | Role |
|---|---|
| Source 1 (S1) | Clean, de-duplicated **reference** list |
| Source 2 (S2) | Noisy fragments to reconcile against S1 |
| Source 3 (S3) | Noisy fragments to reconcile against S1 |

## 3. Task
Entity resolution: for **every S1 entity**, find **all** of its matching records in S2 and S3.
A S1 entity may match many records, exactly one, or **none** (a *singleton*).

## 4. Pipeline the organizers describe
1. **Blocking** — sort records into buckets with a cheap key built from both name and address, so likely
   matches share a bucket. Records can group through a similar name **or** a shared address. Blocking favors
   **recall**: buckets also contain lookalikes (similar name at a different address; different business at
   the same address). Output = **candidate pairs**.
2. **Matching model** — scores each candidate pair, keeps true matches, discards the rest. Output = **final
   matching results**.

Comparing every S1 record with every S2/S3 record is too expensive at scale, hence blocking.
**Blocking sets the ceiling on recall** — a record never considered can never be matched.

## 5. Data
- **Training set**: all three sources plus ground-truth labels.
- **Test set**: the same three sources, no labels; predictions are generated for this.
- **Label format**: one row per S1 entity: `source1_entity_id` and `matched_entity_ids`, a comma-separated
  list of all matching S2/S3 IDs; **empty when the entity matches nothing**. The submission mirrors this.
- **All files are tab-separated** — read with an explicit tab separator (addresses and ID lists contain commas).

## 6. Deliverables
**During the challenge** — upload one file, `matching_results.tsv`: one row per S1 entity with its predicted
matches. It is the **only** file scored on the leaderboard.

**At close** — one archive containing:
- the final matches (`matching_results.tsv`)
- `candidate_pairs.tsv` — the blocking-stage candidate set *before* the model narrowed it (not scored; used to
  audit blocking quality)
- the complete, runnable pipeline
- a methodology document

Top teams' packages are reviewed in detail before final rankings are confirmed.
Run the **provided validation script** before every submission.

## 7. Scoring
**Macro F0.5** over S1 entities — precision weighted 2x over recall:

F0.5 = 1.25 · P · R / (0.25 · P + R)

- A false merge costs roughly twice a missed match; when in doubt, do not merge.
- **Singletons**: predicting an empty list earns **1.0** for that entity; predicting any match earns **0**.
  Identifying non-matching businesses matters as much as finding matches.

## 8. Recommendations and rules
1. Understand singletons.
2. Invest in blocking first (it caps recall).
3. Pay attention to **region-specific patterns** in names and addresses.
4. **Firm rule**: pure ML challenge — **external databases, APIs and lookups are strictly prohibited**. Use
   only the provided data.

## 9. Measured on the data (ours, 2026-09-25; not from the organizers)
_Data obtained from a third-party Hugging Face copy; provenance unverified (see memory notes)._
- Train: 2,206,821 S1 / 5,034,616 S2 / 5,285,603 S3 records. Test: 1,732,544 / 4,887,273 / 5,082,316.
- Fields: `entity_id` (S1-/S2-/S3- prefix), `business_name`, `business_address`, `country`.
- Train countries: US 60%, India 40%. **Test adds France (15% of S1), unseen in train.**
- Every S2/S3 record matches at most one S1 entity; **no match crosses countries**.
- ~26% of S2 and ~25% of S3 records match nothing (distractors). Singletons: 5.58% of S1.
- Matches per S1 entity: 0-11, mean about 3.5 (avg 1.8 from S2, 1.9 from S3).
- Noise seen: typos/digit-for-letter swaps, legal-suffix variants, truncation, website-style names, address
  reordering and abbreviation, corrupted house numbers, missing addresses (~3.4%), names/states in Indic scripts.
- Some ground-truth matches share only the address with S1 (label noise or address-only matches).

## 10. Open question
The video does not define exactly how per-entity precision/recall are computed for an entity with several true
matches (we assume per-entity P and R over the predicted vs. true ID set, then F0.5, averaged over S1 entities).
Confirm against the organizers' scorer or the first leaderboard result.

## 11. Challenge logistics (organizer guidelines, Unstop)
- **Window:** 25 Sep 2026 12:00 AM IST to **27 Sep 2026 11:59 PM IST**.
- **Submissions:** at most **5 per day** over the 3 days (15 total); afterwards the button is disabled.
  Every submission is therefore expensive: tune offline, submit deliberately.
- **Leaderboards:** a **Public** and a **Private** one; evaluation and shortlisting use **both**.
- **Version history:** teams must keep the version history of **all** submissions (shortlisting is based on the
  submitted solutions; final source code may be requested later). See `docs/SUBMISSION_LOG.md`.
- **Artefacts for the best solution:** (a) a **1-2 page document** on the ML approach, models used, experiments and
  conclusion (`docs/METHODOLOGY.md`); (b) **source code** for experiments, training and inference with proper
  comments describing each function.
- **Top 100 teams** must additionally document: methodology, candidate generation / blocking strategy, model
  architecture and feature engineering (covered by docs 03, 05, 06 and `METHODOLOGY.md`).
- **Conduct:** cheating, plagiarism, or unfair practices (e.g. several accounts) lead to instant disqualification.
