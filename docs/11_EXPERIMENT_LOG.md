# 11 — Experiment log: what we tried, what worked, what did not

Every row is a real measurement from this project. "Record-level F0.5" means: on a held-out sample of queries,
treat each query's accepted assignment as one decision and compute F0.5 over those decisions. It is a fast health
check, **not** the leaderboard metric (which is per entity; see 08).

## 11.1 Blocking (recall = true owner among candidates; 100,000 sampled queries per country, full S1 index)

| # | Change | US recall | India recall | Cand./query | Notes |
|---|--------|-----------|--------------|-------------|-------|
| 1 | Token keys + bigrams + squash prefix + address-pair keys, one cap (40) | 97.8 % | – | 25 | ≈ 55 % of misses had an *empty address* |
| 2 | Separate caps per key type (bigram/squash 150, pairs 100) | 98.35 % | 94.2 % | 28 | India: 1,971 of 4,267 misses were native-script names |
| 3 | + `h:` number×word keys; wider S1 side for `p:` keys | 98.35 % | 96.0 % | 28–29 | native-name recall 85 → 90 % |
| 4 | + leading-zero removal, + `x:` sorted-address key | 98.45 % | 96.0 % | 28 | small gains |
| 5 | + native aliases | 98.3 % | 96.4 % | 29 | blocking gain small; the big alias gain is in the matcher |

## 11.2 Matcher (record-level F0.5 on held-out fold)

| # | Change | F0.5 | Precision | Recall | Accepted distractors |
|---|--------|------|-----------|--------|----------------------|
| 1 | Baseline: similarities + blocking evidence, 150 k queries/country | 0.9747 | 99.01 % | 91.75 % | 351 |
| 2 | + **leftover token / number features** | 0.9787 | 99.28 % | 92.61 % | 254 |
| 3 | + all aliases applied (first attempt) | 0.9790 | 99.15 % | 93.19 % | 293 |

Alias-mode A/B on *identical* queries (this is why we retest instead of trusting a single run):

| Mode | F0.5 | US recall | India recall |
|------|------|-----------|--------------|
| none | 0.9777 | 94.90 % | 90.48 % |
| all | 0.9782 | 94.37 % ↓ | 92.03 % |
| native_addr | 0.9809 | 94.82 % | 93.24 % |
| **native** | **0.9820** | 94.94 % | 93.62 % |

| # | Change | F0.5 | Precision | Recall |
|---|--------|------|-----------|--------|
| 4 | `native` mode, 150 k queries/country | 0.9820 | 99.33 % | 93.94 % |
| 5 | 300 k queries/country, `min_data_in_leaf` 200 | **0.9827** | 99.37 % | 94.09 % |

Where the remaining recall goes (5): ≈ 92.7 % of matched queries accepted; 2.4 % never reached by blocking;
5.0 % reached but rejected (2.5 % with p < 0.3 — mostly empty-address or native-script records with little
evidence; 2.1 % in the ambiguous 0.3–0.7 zone; 0.4 % lost to a near-tie with another entity).

## 11.2b Decoding experiments (entity-level macro F0.5, full US+India train world, held-out entities)

| # | Decoder | F0.5 | Singletons | Notes |
|---|---------|------|------------|-------|
| 1 | Threshold `p1 ≥ 0.7` | 0.97558 | 0.9725 | baseline; best of a grid over t, gap, entity guard |
| 2 | Expected-F0.5 per entity (calibrated) | 0.97559 | 0.9493 | **tie** — gained recall, lost the same on singletons; kept in code, not used |
| 3 | **Entity-context second stage, `r ≥ 0.7`** | **0.97649** | 0.9836 | +0.00091; used for the current best submission file |

Loss analysis that motivated (3): 74 % of the entity-level loss is *records missed inside entities we already found* (336 k entities lose only
recall, 35 k add only a wrong record), not false merges (see 08, 8.5.2).

## 11.3 Things that did NOT help (and were kept out)

| Idea | Result |
|------|--------|
| Rewriting **Latin** abbreviations (`texas → tx`, `pvt → private`, `st → street`) in queries | slightly *worse* for the US; the fuzzy features already absorb it → removed from default |
| Requiring the address to match exactly | only ≈ 9 % of true pairs match exactly → would destroy recall |
| Expected-F0.5 decoder (as a replacement for thresholding) | tied the baseline (+0.00001): its calibrated probabilities cannot separate a genuine lone record from a decoy attached to a singleton (07) |
| Using `unidecode` for transliteration | removed: a lookup table is an external-resource risk under the rules |

## 11.4 Bugs and surprises found on the way (so nobody repeats them)

| Discovery | Consequence |
|-----------|-------------|
| Parquet stores missing addresses as the string `"nan"`, TSV as empty | Use TSV only |
| An empty match list is `null` in the Parquet ground truth, `""` in TSV | Filter both (`notna & != ""`) |
| A validation script that only counted queries *with candidates* silently ignored blocking misses | Denominators now include every sampled query |
| `pkill -f` matched its own shell command | Kill by PID / avoid pattern-killing inside the same command |
| NTFS marked freshly created folders read-only mid-session | `chmod u+w` (see 10) |
| Polars' CSV writer quotes empty strings, so empty match lists were written as a literal `""` (102,695 rows) | Would likely have been read as a match id or rejected. Fixed with `write_submission` (quoting off) plus a byte-exact test and a post-write assertion; caught before any upload |
| A vacuous test assertion (`... or True`) | Rewritten to assert the real behaviour |

## 11.5 Open items and ideas not yet done
* **Looser blocking caps.** 2.14 % of true matches have no plausible candidate; ≈ 55 % of those are non-empty, non-native records that
  blocking dropped because a key exceeded its frequency cap. The experiment (`eval_blocking.py --key_cap 100/250 --bigram_cap 400/800
  --pair_cap 300/600`) was started but interrupted by a session end, so **no result exists yet**. It is the most promising recall lever.
* Character-level fuzzy blocking keys (e.g. deletion neighbourhoods) for typos in the only informative token.
* Word segmentation of glued names (`mediajayaindia` → `media jaya india`) using the learned vocabulary.
* Entity-context second-stage features (how consistent an entity's S2 and S3 records are with each other).
* Transductive self-training for France (see 09).
