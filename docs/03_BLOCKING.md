# 03 — Blocking: finding plausible candidates without comparing everything

**Code:** `src/entity_resolution/blocking.py` · **Script:** `scripts/eval_blocking.py`

## 3.1 The problem this solves

For the test set we have ≈ 10 million queries (S2 + S3 records) and ≈ 1.7 million entities (S1). Comparing every
query with every entity means ≈ 17 *trillion* comparisons — impossible. **Blocking** is a cheap first pass that
picks, for each query, a short list of entities that *might* be its owner. The expensive model (06) then only
looks at that short list.

The organizers put it plainly: **blocking sets the ceiling on recall.** If the true owner is not in the short
list, nothing downstream can recover it. So blocking is designed to be *generous*: it may include wrong entities
(the model will reject them) but should almost never miss the right one.

**Input:** the cleaned records of one country (`S1` as the index; `S2+S3` records as queries).
**Output:** for each query, up to 30 candidate entities with some evidence scores (`n_keys`, `score`, …).

## 3.2 The idea in plain language

Think of a library where every book gets several **tags** (keywords). To find likely duplicates of a book you do
not read every other book; you look at books that share a *rare* tag. A tag like "the" is useless because half the
library has it. A tag like "plainfield" or "1208" is informative because few books have it.

We do exactly this:

1. Turn every record (entity or query) into a set of **keys** (tags).
2. **Index** the entities: for each key, which entities have it, and how many (the *document frequency*, "df").
3. **Discard unselective keys**: a key shared by more than *cap* entities is ignored (it would pull in too many
   unrelated candidates).
4. For each query, look up each of its keys; every entity found gets **evidence**. Weight each shared key by its
   rarity (**IDF** = log(N / df): rare key → big weight) and add up.
5. Keep the **top 30** entities by total evidence.

Because keys come from *both* the name and the address, a query can be found through a similar name, through the
same address, or both — exactly as the organizers describe.

## 3.3 The keys (tag types)

Each key is written `type:value`. A record produces all of the following that apply:

| Type | Built from | Example | Why it exists | Cap (max entities sharing it) |
|---|---|---|---|---|
| `n:` | each name token | `n:oncology` | plain name overlap | 40 |
| `a:` | each address token | `a:plainfield` | plain address overlap | 40 |
| `b:` | each pair of adjacent name words | `b:cardiology_specialists` | more specific than single words; helps with common words | 150 |
| `q:` | first 8 letters of the *squashed* name | `q:kimbleol` | matches glued/domain-style names to spaced names | 150 |
| `p:` | pairs among the record's rarest address tokens | `p:firehouse|willis` | a record sharing two rare address words is very likely the same place | 100 |
| `h:` | (number token, word token) pairs from the address | `h:1208|hyderabad` | survives address **truncation** (see 3.5) | 100 |
| `x:` | the address's tokens, sorted, joined | `x:c d ... noida` | order-insensitive whole-address match: catches complete but reordered addresses | 150 |

Notes on the two-sided design of `p:` keys: on the S1 side we take the 7 rarest address tokens, on the query
side the 5 rarest. They are asymmetric on purpose — a query with a shortened address might keep only the tokens
that are *common* in S1's longer address, and we still want the pair to line up.

## 3.4 How the scoring and pruning work

For each (query, entity) pair we accumulate: `n_keys` (how many keys they share), `score` (sum of IDF of the
shared keys), `max_idf` (the strongest single key), and one counter per key type. Pairs are sorted by `score` and
the top **30** per query are kept. Those numbers are later reused as features by the matcher (05) — they are
genuinely informative (the single most important feature of the matcher is `score`).

**Cost:** everything is a polars join/group-by; nothing is quadratic. Blocking ≈ 16 seconds per 100,000 queries
on this machine; the S1 index is built **once per country** (`build_index`) and reused for every chunk of queries
(`query_candidates`).

## 3.5 Why each key type exists (the misses that motivated them)

We measured recall by sampling queries and checking whether the *true* entity (from the training labels) was in
the candidate list, then reading through the misses. Each key type was added to fix a category of misses:

| Miss we saw | Fix | Effect |
|---|---|---|
| Common words made name-only matches impossible | separate, larger caps for the more selective bigram/squash keys | US 97.8 % → 98.4 % |
| Native-script records whose address was *shortened* to `plot no 1208 hyderabad telangana` while S1 had a long address. Rare-token pair keys on each side never lined up (S1's rare tokens were `spline`, `arcade`; the query kept only common ones) | `h:` number×word keys, wider S1 side for `p:` | India 94.2 % → 96.0 % |
| Complete but reordered / identical address, different name | `x:` sorted-address key | small extra gain |
| `0019553` vs `19553` | leading-zero stripping in normalization (02) | small extra gain |

## 3.6 What blocking achieves (measured on train, full S1 index, 100,000 sampled queries per country)

| Country | Recall (true entity in candidates) | Candidates per query |
|---|---|---|
| US | **98.4 %** | ≈ 28 |
| India | **96.0–96.4 %** | ≈ 29 |

What is still missed, and why we accepted it:

* **Empty-address queries whose name is common** (≈ 55 % of US misses; recall for empty-address queries is
  ≈ 80 %). With no address, a common name like `urology health inc` fits dozens of entities; no algorithm can
  pick one, and a precision-first system should not guess anyway.
* **Native-script names with almost no address left** (India recall for such records ≈ 90 %). Same reasoning.
* **Typos in the only informative tokens.** Would need character-level fuzzy keys; measured as a small
  remainder and left as future work.

## 3.7 Design decisions and rejected alternatives

* **Rejected: fixed "blocks" (sort by a single key).** One key breaks under any noise in that key. Multiple
  independent keys give redundancy.
* **Rejected: embeddings / neural nearest-neighbours.** Slower, harder to explain, no better recall here, and
  would pull in pretrained models (a rules risk).
* **Rejected: all-pairs within a country.** Quadratic — would take days.
* **Country partitioning is free:** matches never cross countries (01), so each country is blocked
  separately.

## 3.8 `candidate_pairs.tsv` (one of the required deliverables)

The organizers audit blocking quality using this file. `score_country(..., cand_dir=...)` can write every
candidate pair (query id, entity id, evidence score) per chunk while scoring. The exact column layout the
organizers expect is unspecified in the video; we assume `source1_entity_id<TAB>candidate_entity_id`. The size
is large (≈ 28 pairs per query × 10 M queries), see `10_RUNNING_THE_PIPELINE.md` for how the final file is
produced.

## 3.9 Hand-off
The candidate list, together with the evidence numbers (`score`, `n_keys`, …), goes to **pair features** (05).
