# 05 — Pair features: turning "two records" into numbers

**Code:** `src/entity_resolution/features.py` · **Throughput:** ≈ 0.3 million pairs per second

## 5.1 The problem this solves

A machine-learning model cannot read two strings; it needs a row of numbers. For every candidate pair
(query, entity) we compute ≈ 60 numbers that describe *how* the two records agree and disagree. The matcher (06)
learns which combinations of numbers indicate "same business".

**Input:** candidate pairs from blocking + the cleaned text of both records.
**Output:** one row of features per pair, all numeric, and none of them depend on which country the pair is from
(important for France — see 09).

## 5.2 The feature groups

### A. Blocking evidence (from `03`)
`score`, `max_idf`, `n_keys`, and per-key-type counts `k_name`, `k_addr`, `k_bigram`, `k_sq`, `k_pair`,
`k_numpair`, `k_addrbag`. *Meaning:* how much rare evidence the two records share. Blocking's `score` is the
strongest single feature of the model.

### B. Name similarity
| Feature | What it measures |
|---|---|
| `n_ratio` | overall character similarity of the two cleaned names (0–100) |
| `n_tsort` | similarity ignoring word order (`"acme robotics"` vs `"robotics acme"`) |
| `n_tset` | similarity that forgives extra words (`"acme"` vs `"acme robotics inc"`) |
| `n_jw` | Jaro-Winkler: friendly to typos, rewards a matching beginning |
| `n_exact` | names identical? |
| `sq_ratio`, `sq_partial` | the same comparisons on the *squashed* names, so `kimbleolva` (a domain-style name) can match `kimble olva` |

### C. Address similarity
`a_ratio`, `a_tsort`, `a_tset`, `a_partial` (partial token set: does one address contain the other?), `a_jw`,
`a_exact`.

### D. Numbers (house numbers and other digits)
Numbers are the most discriminating part of an address, and the noise generator attacks them specifically.
* `num_inter` — how many number tokens the two addresses share; `num1`, `num2` — how many each has.
* `num_first_eq` — is the first number the same?
* `num_ratio` — similarity of the whole "numbers only" strings (`"914 39"` vs `"91 39"` is high: one digit lost).

### E. The **leftover** features (the key innovation, see 1.4)
After removing the tokens two records share, look at what is *left* on each side:

```
S1: merna    santacruz classic stardust pc         leftover S1 = "merna"
Q : dittmer  santacruz classic stardust pc         leftover Q  = "dittmer"      → very different → suspicious
S1: tejansh  clinical llp                          leftover S1 = "tejansh"
Q : tejanshyn clinical llp                         leftover Q  = "tejanshyn"    → nearly the same → just a typo
```
Features: `n_rest_ratio` (similarity of the two leftover strings), `n_rest_len1/len2` (how much is left), and the same
for the address (`a_rest_*`) and for **house-number-like tokens** (`m_rest_ratio`, `m_rest_len1/2`,
`m_rest_dist` = edit distance between leftover numbers: `11788` vs `11797` is a *substitution of the same
length*, whereas a lost digit `914 → 91` is a *deletion*). Also `n_shared_tok`, `a_shared_tok`, token counts.

**Why this is powerful:** the dataset's distractors are made by changing *one word* or *a few digits* of a real
entity. Plain similarity sees "95 % the same" and is fooled; leftover features see that the 5 % is a
*substitution*, not noise. Adding this group cut accepted distractors by ≈ 28 % and lifted record F0.5
0.9747 → 0.9787.

### F. Ambiguity of the reference
`s1_name_dup`, `s1_addr_dup`: how many S1 entities in the country share this exact name / address. If 20 entities
are called `urology health inc`, a name match means little; if only one is, it means a lot.

### G. Record-quality flags
`q_addr_empty`, `q_is_domain`, `src` (S2 or S3 — the two vendors have different noise styles), lengths and length
ratios (`n1_len`, `n2_len`, `a1_len`, `a2_len`, `n_len_ratio`, `a_len_ratio`), and the native-script flags from
04 (`q_native_name`, `q_native_addr`, `q_unmapped_name`, `q_unmapped_addr`).

## 5.3 Missing values
If a record has no numbers, features like `num_first_eq` are *missing* (NaN), not zero. LightGBM (06) treats
"missing" as its own signal, which is exactly right: "no number to compare" differs from "numbers differ".

## 5.4 Implementation notes (for speed)
* String similarities run in **rapidfuzz's multithreaded C++** through `process.cpdist`, which scores two
  parallel lists of strings pairwise — ≈ 100× faster than Python loops.
* Token/number set operations run as vectorized **polars** list expressions.
* Work is done in chunks of 1.5 M pairs to bound memory. The S1 side (with its ambiguity counts) is prepared once
  per country (`prepare_s1`).
* No feature reads the country, so nothing is tuned to US/India.

## 5.5 Hand-off
The feature table goes to the matcher (06), in the exact column order saved in `matcher_features.json`.
