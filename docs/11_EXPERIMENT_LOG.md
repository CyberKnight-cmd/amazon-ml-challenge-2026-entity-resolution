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
| 6 | Baseline re-measured 27 Sep (same sample, aliases from `artifacts/`) | 98.44 % | 96.83 % | 28–29 | reference for row 7 |
| 7 | **+ v2 generator-aware keys** (v1 top-30 ∪ v2 top-15; 03, 3.9) | **99.41 %** | **98.52 %** | 38–40 | recovers 53–62 % of v1 misses; v2 alone at top-30: 98.86 % / 97.23 % |
| 7a | First v2 prototype, v1 ∪ v2 top-10 (names made only of frequent words skipped, no alias classes, out-of-vocabulary words allowed as "rare", no `sw` keys) | 99.18 % | – | 34 | the same union with the fixes of row 7: 99.35 % at 34 → fixes worth +0.17 pt (US) |

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
| 3 | Entity-context second stage, `r ≥ 0.7` | 0.97649 | 0.9836 | +0.00091 |
| 4 | Stage-2 re-ranker (noisy-channel + context + vocab + house-number features), `r ≥ 0.6` | 0.98473 | 0.9931 | `stage2-v1`, public LB **0.98743** |
| 5 | Stage-1 threshold `p1 ≥ 0.7`, **v2 blocking** (v1 top-30 ∪ v2 top-15, v2 evidence as features) | 0.98061 | 0.9757 | +0.0050 over row 1 from blocking alone |
| 6 | Stage-2 on v2 blocking + **sibling-agreement features**, `r ≥ 0.6` | 0.98882 | 0.9931 | `v2blk-sib`; +0.0041 over row 4 (P 0.9980, R 0.9716 vs 0.9979, 0.9609) |
| 7 | **+ cross-encoder score** on stage-1 `p ≤ 0.98` pairs, `r ≥ 0.6` | **0.98934** | 0.9950 | `v2ce-hard`; +0.0005 over row 6 although stage 2 is fitted on ⅔ of half A (the rest trained the cross-encoder) |

Loss analysis that motivated (3): 74 % of the entity-level loss is *records missed inside entities we already found* (336 k entities lose only
recall, 35 k add only a wrong record), not false merges (see 08, 8.5.2).

Loss analysis of row 4 on held-out true pairs (27 Sep): 96.02 % accepted; 2.67 % never reached stage 2 (mostly blocking, which motivated
row 5); 0.69 % lost to another entity; 0.64 % rejected. About half of all misses are **empty-address records**, and ~1.5 % of true pairs are
empty-address records whose S1 name is shared by 2–10+ entities; there is no text to separate those, for anyone.

Sibling agreement (rows 6–7): each source's copies of an entity are noised from one per-source variant (an S3 house number 4210 where S1 says
4211, an added locality), so on hard pairs (0.05 < p < 0.95) **53 % of true matches but 7 % of decoys** repeat a rare divergence from S1 that
one of the entity's confident records also has.

Cross-encoder (row 7): `intfloat/multilingual-e5-small` (MIT, 118 M) fine-tuned on 310 k hard pairs of ⅓ of stage-2's fitting half,
checked on 60 k hard pairs of the held-out half:

| | AUC | log-loss | AUC combined with stage-1 p |
|---|---|---|---|
| stage-1 p | 0.9160 | 0.3393 | – |
| cross-encoder, 1 epoch | 0.9458 | 0.2721 | 0.9650 |
| 2 epochs | 0.9531 | 0.2546 | 0.9686 |
| **3 epochs (used)** | **0.9555** | **0.2480** | **0.9699** |

## 11.3 Things that did NOT help (and were kept out)

| Idea | Result |
|------|--------|
| Rewriting **Latin** abbreviations (`texas → tx`, `pvt → private`, `st → street`) in queries | slightly *worse* for the US; the fuzzy features already absorb it → removed from default |
| Requiring the address to match exactly | only ≈ 9 % of true pairs match exactly → would destroy recall |
| Expected-F0.5 decoder (as a replacement for thresholding) | tied the baseline (+0.00001): its calibrated probabilities cannot separate a genuine lone record from a decoy attached to a singleton (07) |
| Using `unidecode` for transliteration | removed: a lookup table is an external-resource risk under the rules |
| **Sibling-found candidates**: propose the entity of a same-source confident record sharing an address-variant key | on the wB world 1.12 M new pairs of which 0.5 % true, recovering 0.22 % of true pairs → off (`stage2_build.py --sibling_cands`); the signal works as a *feature* (row 6), not as a candidate generator |
| Punctuation marks to separate S1 entities with the same normalized name | 164 k of 181 k same-name groups have identical raw names → at most ~10 % of those misses reachable; not built |
| Row-order or id-number leakage | checked: Spearman ≈ 0 between S1 and matched record positions / id numbers — no leak |

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
* ~~Character-level fuzzy blocking keys (e.g. deletion neighbourhoods) for typos in the only informative token.~~
  Done as part of blocking v2 (row 7 of 11.1), together with token-level deletion neighbourhoods, which turned
  out to matter far more than character-level ones.
* Word segmentation of glued names (`mediajayaindia` → `media jaya india`) using the learned vocabulary.
* Entity-context second-stage features (how consistent an entity's S2 and S3 records are with each other).
* Transductive self-training for France (see 09).
