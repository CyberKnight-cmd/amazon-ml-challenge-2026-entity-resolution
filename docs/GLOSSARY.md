# Glossary

Terms are listed alphabetically. Where a term has a home document, it is given in brackets.

**Alias table** — A list "query token → S1 token" learned from training pairs (`लिमिटेड → limited`). [04]

**Blocking** — A cheap first pass that proposes, for each query, a short list of plausible entities so that the
expensive model does not have to compare everything with everything. [03]

**Calibration** — Turning a model's raw score into a *trustworthy probability*, i.e. among all pairs scored 0.9,
about 90 % are truly matches. Needed by the expected-F0.5 decoder. [07]

**Candidate pair** — A (query, entity) pair proposed by blocking. `candidate_pairs.tsv` lists them. [03]

**Distractor** — An S2/S3 record that matches *no* S1 entity (≈ 25 %). Many are deliberately built as
near-duplicates of real entities. [01]

**Document frequency (df)** — How many S1 entities contain a given key. Rare keys (small df) are informative. [03]

**Entity** — A Source 1 business (the clean reference). The leaderboard scores one F0.5 per entity. [PROBLEM_STATEMENT]

**F0.5** — The metric: `1.25·P·R / (0.25·P + R)`. Precision counts about twice as much as recall. [PROBLEM_STATEMENT]

**Feature** — One number describing a pair (e.g. name similarity 87, house numbers equal = 1). [05]

**Fold** — A chunk of entities held out for validation; assigned by a hash of the entity id so no entity is in
both training and validation. [06]

**Gap** — `p1 − p2`: how much better a query's best candidate is than its runner-up. Small gap = ambiguous. [07]

**IDF** — Inverse document frequency, `log(N/df)`: the weight of a key by its rarity. [03]

**Leftover tokens** — Words that remain in each record after removing the words both share; used to tell typos
from substituted words. [05]

**LightGBM** — The gradient-boosted decision-tree library used for the matcher. [06]

**Macro-average** — Average of per-entity scores, each entity counting equally, however many records it has. [08]

**Native script** — Text written in an Indian script (Devanagari, Tamil, …) instead of Latin letters. [04]

**Normalization** — Cleaning text (lowercase, no accents/punctuation, etc.) before comparing. [02]

**p, p1, p2** — Matcher probability of a pair; the best and second-best probability for one query. [06, 07]

**Precision / Recall** — Precision = share of predicted matches that are right; recall = share of true matches
found. [PROBLEM_STATEMENT]

**Query** — An S2 or S3 record that we try to assign to an entity (or to nobody). [01]

**Singleton** — An S1 entity with no matching record; predicting any match for it scores 0, predicting an empty
list scores 1. [01, 07]

**Squashed name** — The name with spaces removed, used to compare glued domain-style names. [02]

**Threshold decoder** — Accept a query's best candidate if `p1 ≥ t` and `p1 − p2 ≥ gap`. The simple baseline
decoder. [07]

**Expected-F0.5 decoder** — Decoder that picks, per entity, the set of records with the highest *expected*
score, including the option of predicting nothing. [07]

**World** — The set of all entities and queries considered together. Evaluating on the *full* train world
(all queries, all entities of a country) reproduces the leaderboard's conditions. [08]
