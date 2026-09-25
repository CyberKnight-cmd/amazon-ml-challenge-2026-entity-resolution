# 04 — Learned aliases: matching Indian-script names without a dictionary

**Code:** `src/entity_resolution/aliases.py` · **Script:** `scripts/fit_aliases.py` · **Tests:**
`tests/test_pipeline_pieces.py`

## 4.1 The problem this solves

In India, about 23 % of S2 names and 13 % of S3 names are written in native scripts:

```
S1 : anand premier it limited
Q  : आनंद प्रीमियर आईटी लिमिटेड        ← the same name, Devanagari script
```

To a program these strings share **no characters at all**, so every name-similarity feature is zero and the
matcher concludes "different name → probably a look-alike distractor". In one of our error analyses a perfectly
matching record (identical address!) got probability **0.01** for exactly this reason.

We need name evidence for those records. The obvious tool — a transliteration dictionary or library — is
exactly what the rules forbid ("external databases, APIs and lookups are strictly prohibited"). So we **learn the
mapping from the training data itself**.

## 4.2 The idea in plain language

We have millions of *known* pairs (S1 entity, matching query) from the training labels. In a pair where the query
name is in Hindi and the S1 name is in English, the two names are the same name written twice. If we line up their
words position by position we get translation examples:

```
S1 :  anand     premier      it     limited
Q  :  आनंद       प्रीमियर      आईटी    लिमिटेड
```

One pair gives four examples. Across a million pairs the word `लिमिटेड` lines up with `limited` **38,118 times out
of 38,118** — a certain translation. The same works for `प्राइवेट → private` (31,465 / 31,465), `टेक → tech`,
`इंटरनेशनल → international`, `ग्लोबल → global`, `आईटी → it`…  Nobody typed a dictionary; we *counted*.

## 4.3 The algorithm (`fit_aliases`)

For each labelled pair we compare the query's tokens with the S1 record's tokens:

1. **Remove the tokens both sides share.** What is left are the "leftover" tokens on each side.
2. If both sides have **the same small number of leftovers (1–4)**, align them **in order** and record each
   (query token → S1 token) pair. (Equal counts + order = high confidence; otherwise we skip the pair rather than
   guess.)
3. **Count** each (query token, S1 token) alignment over ~1.5 million pairs.
4. For each query token keep its **dominant partner** if it is reliable: seen at least 5 times and the top
   partner accounts for at least 60 % of the sightings ("purity"). Tokens that are already identical are ignored.

This is done separately for the **name** field and the **address** field. Result: a table of ≈ 6,500 aliases
(3,096 for names, 3,415 for addresses), stored in `aliases.parquet`.

**No leakage:** the table is fitted only on pairs whose S1 entity is *not* in validation fold 0, so validation
numbers are not contaminated.

## 4.4 What it learned (real output)

| Query token | → S1 token | Seen | Purity |
|---|---|---|---|
| `लिमिटेड` | `limited` | 38,118 | 100 % |
| `ltd` | `limited` | 33,496 | 99.6 % |
| `प्राइवेट` | `private` | 31,465 | 100 % |
| `incorporated` | `inc` | 8,036 | 99.8 % |
| `st` (address) | `street` | 41,847 | 86 % |
| `texas` (address) | `tx` | 26,837 | 89 % |
| `ka` (address) | `karnataka` | 5,142 | 96 % |

Interesting by-product: the algorithm *discovered* that S1 spells this legal form `inc` (not `incorporated`) and
`limited` (not `ltd`) — it learns **S1's spelling conventions**, because we always map *towards* S1.

## 4.5 Applying it (`apply_aliases`) — and the experiment that told us how much to apply

Applying = rewriting each **query** token that has an alias to the S1 spelling, then recomputing the squashed
name. We tested four modes on the *same* validation queries (record-level F0.5; see `11_EXPERIMENT_LOG.md`):

| Mode | What is rewritten | F0.5 | US recall | India recall |
|---|---|---|---|---|
| `none` | nothing | 0.9777 | 94.9 % | 90.5 % |
| `all` | every alias, names and addresses | 0.9782 | 94.4 % ↓ | 92.0 % |
| `native_addr` | native-script name tokens + all address aliases | 0.9809 | 94.8 % | 93.2 % |
| **`native`** | **only native-script tokens** | **0.9820** | 94.9 % | **93.6 %** |

**Lesson:** the cross-script aliases are the valuable ones. Rewriting Latin abbreviations (`texas → tx`) did
*not* help and slightly hurt the US — the fuzzy string features already absorb those small differences, and a
rewrite can occasionally destroy a distinguishing detail. So the default is `native` (`ER_ALIAS_MODE=native`,
`dataset.alias_mode()`).

## 4.6 Native-script flags
Before rewriting, `add_native_flags` records whether the name / address *originally* contained Indic script
(`native_name`, `native_addr`). The matcher gets these flags (05) so it knows that the name evidence for this
record came through a rewrite and how much to trust it. It also gets `q_unmapped_name/addr`: after rewriting,
does any native-script text remain? (Rare names never seen in training stay unmapped.)

## 4.7 Limits and honest caveats

* **Coverage is limited to words seen in training.** Very rare names stay in native script. Blocking still finds
  many of those records through the (Latin) address.
* **It cannot help France.** France is not in the training set, so no French aliases can be learned; and French
  is written in Latin script anyway, so we do not need cross-script rewriting there (09).
* **Order-based alignment** assumes roughly monotonic word order; the purity threshold and equal-count filter keep
  garbage alignments out of the table.

## 4.8 Hand-off
Rewritten queries go to blocking (03) and feature computation (05). The alias table is a fitted artifact saved
next to the model (see 10).
