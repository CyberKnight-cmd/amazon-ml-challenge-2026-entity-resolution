# 02 — Normalization: making "the same" strings look the same

**Code:** `src/entity_resolution/normalize.py`, `io.py` · **Script:** `scripts/build_interim.py` ·
**Tests:** `tests/test_normalize.py`

## 2.1 The problem this solves

Computers compare characters, not meaning. `"PLAINFIELD RD."`, `"Plainfield Road"` and `"plainfield road"` are
the same street to a human but three different strings to a program. Before any comparison we clean both fields
so that harmless differences (case, punctuation, accents, stray placeholders) disappear, and only *real*
differences remain.

**Input:** raw `business_name`, `business_address` (strings, arbitrary noise).
**Output:** for every record, cleaned `name_n`, `addr_n`, a `name_sq` (name without spaces), and an
`is_domain` flag. Cached in `data/interim/<split>_s<source>.parquet` so we clean each file once (≈ 75 s for all
26 M records, using 12 processes).

## 2.2 What we do, step by step

### Names (`norm_name`)
1. **Lowercase and strip Latin accents.** `Collège` → `college`, `LÍBERTY` → `liberty`. We only strip *Latin*
   combining marks (range U+0300–U+036F) and handle ligatures such as `ß`, `æ`, `œ`. **Indian scripts are left
   untouched** — their vowel signs (matras) are real letters; deleting them would destroy the words.
2. **`&` → `and`.** `Brown & Warfield` = `Brown and Warfield`.
3. **Punctuation → space.** We keep digits, `a–z`, and the Indic block U+0900–U+0DFF. (Python's `\w` would drop
   Indic vowel signs, which is why we use an explicit character class.)
4. **Collapse repeated adjacent tokens.** Noise sometimes duplicates a word: `Heritage Heritage Semiconductor`,
   `CARDIOLOGY SAFE CARE INC [INC]` → `cardiology safe care inc`.
5. **Domain-style names.** A name with no spaces that looks like a website (`kimb1eolvas.com`,
   `cardiology-safe-care.com`) becomes one token with the top-level domain and hyphens removed
   (`kimb1eolvas`, `cardiologysafecare`) and `is_domain = True`. These records are ≈ 3–4 % of S2/S3 and their
   words are glued together, so ordinary word matching cannot see inside them (handled later by the
   squashed-name features).

### Addresses (`norm_addr`)
1. Same lowercase / accent / punctuation handling.
2. **Drop "missing" components.** The address is split on commas; pieces that mean "no value"
   (`N/A`, `NULL`, `nan`, `none`, `-`) are removed. Example: `HOMELAND AVE, NULL, NORMAN, OK` →
   `homeland ave norman ok`.
3. **Strip leading zeros from numbers.** `0019553` → `19553`, `003651` → `3651`. Some records pad house numbers
   with zeros; before this fix such pairs shared *no* number token (this was one of the misses we found in
   blocking analysis).

### The squashed name
`name_sq` = the cleaned name with spaces removed (`kimble olva` → `kimbleolva`). It lets us compare a
spaced name against a glued domain-style name.

## 2.3 What we deliberately do NOT do

* **No hand-written abbreviation tables** (`st → street`, `tx → texas`, `pvt → private`). Two reasons:
  (a) the rules forbid external lookups and a hand-typed table is dangerously close to one; (b) the test set
  contains **France**, which has different abbreviations we could never list in advance. Instead we *learn*
  aliases from the training data ([04](04_CROSS_SCRIPT_ALIASES.md)), and found that only the cross-script ones
  actually help.
* **No country-specific rules.** Nothing in this file mentions India or the US. That is what makes it safe for
  the unseen France data.
* **No transliteration library.** A package like `unidecode` is effectively a lookup table; we removed it from
  the dependencies for that reason.

## 2.4 How it is executed at scale

`io.py` reads the TSV with polars (`infer_schema=False`, `quote_char=None`, because `"` inside names is data,
not CSV quoting), then normalizes in chunks of 250,000 rows across a process pool, and stores the result as
zstd Parquet. All later stages read these cached files.

## 2.5 Tests (what is guaranteed)

`tests/test_normalize.py` checks: accent stripping keeps Indic vowel signs; `&` handling; duplicate-token
collapse (including the `[INC]` case); domain-style detection and its negative case (`Acme Robotics.com` has a
space so it is *not* a domain); missing-component removal; Indic tokens surviving in addresses; leading-zero
stripping; squashing.

## 2.6 Hand-off to the next stage
Cleaned text goes to **cross-script aliasing** (only for queries) and then to **blocking**. Both assume the
exact conventions above (lowercase, space-separated tokens, no punctuation), so any change to this file
requires rebuilding the interim files (`scripts/build_interim.py`) and retraining.
