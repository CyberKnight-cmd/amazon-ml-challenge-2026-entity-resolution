# 01 — The data and what we discovered about it

_Read this first: every design decision in the later documents traces back to a finding here._

## 1.1 The files

Everything lives under `data/raw/` (never committed to git — see `.gitignore`). Both **TSV** (the official
format) and **Parquet** (a faster copy) exist. **We use the TSVs**: the Parquet copy stores a missing address as
the literal text `"nan"` while the TSV has an empty field, so the two are *not* byte-for-byte equivalent.

| File | Rows | Meaning |
|------|------|---------|
| `train_source1` | 2,206,821 | Clean reference businesses (S1) |
| `train_source2` | 5,034,616 | Noisy records from vendor 2 |
| `train_source3` | 5,285,603 | Noisy records from vendor 3 |
| `train_ground_truth` | 2,206,821 | One row per S1 business: the comma-separated list of its S2/S3 records (empty = singleton) |
| `test_source1/2/3` | 1,732,544 / 4,887,273 / 5,082,316 | Same, but **no labels** |

Every source file has the same four columns: `entity_id`, `business_name`, `business_address`, `country`.
IDs carry their source as a prefix (`S1-…`, `S2-…`, `S3-…`).

> **Terminology.** Throughout the docs a **query** is an S2 or S3 record (the noisy thing we are trying to
> place), and an **entity** is an S1 business (the clean thing it might belong to). We ask, for each query:
> *which entity, if any, is it a noisy copy of?*

## 1.2 Facts we measured (all verified on the real files)

| Finding | Number | Why it matters |
|---|---|---|
| A query matches **at most one** entity | 0 records with two owners | We can treat the task as "each query picks one entity or none" (see 07) |
| Matches **never cross countries** | 0 cross-country pairs | We can run everything per country — smaller problems, zero recall loss |
| Queries that match nothing (**distractors**) | ≈ 26 % of S2, ≈ 25 % of S3 | The model must learn to say "none of these" |
| **Singleton** entities | 5.58 % of S1 | Predicting *any* record for them scores 0, so wrong guesses are very expensive |
| Matches per entity | 0–11, mean ≈ 3.5 | Entities usually have several records, e.g. 2 from S2 and 2 from S3 |
| Train countries | US 60 %, India 40 % | – |
| **Test countries** | India 47 %, US 38 %, **France 15 %** | France never appears in training — see 09 |
| Missing addresses | ≈ 3.4 % of S2/S3 records | Name-only evidence; often unresolvable |
| S1 names that occur more than once | ≈ 30 % | Same name at different addresses → name alone cannot decide |

## 1.3 How the noise looks (real examples)

These are actual matched records (normalized for readability). Notice how many different corruptions exist.

```
S1  : oncology physicians            | 914 39th street, austin, tx
S2  : Oncology Physicians Corporation | TX, AUSTIN, 914 39TH ST                      ← reordered, abbreviated, extra word
S3  : Oncology  Physicians            | 91 39th Saint, Austin, Texas                 ← lost a digit, "Saint" for "Street"
S3  : oncologyphysicians.com          | Austin, Texas, 91 39th Street                ← website-style name

S1  : Cardiology Safe Care Inc        | 617 Firehouse Road, VA, Floyd County
S2  : Cardiology 5afe Cáre Inc        | 617 FIREHOUSE ROAD, FLOYD COUNTY, VA         ← digit-for-letter typo, accent
S3  : CARDIOLOGY SAFE CARE INC [INC]  | 617 Firehouse Rd, Floyd County, Virginia     ← duplicated suffix, full state name

S1  : Al Tech Private Limited         | The Ambidence Court Plot No. 2, ... Vashi, Maharashtra
S2  : अल टेक प्राइवेट लिमिटेड              | 1605/ 1606, THE AMBIDENCE COURT ... VASHI, Maharashtra   ← name in Devanagari script
```

The corruptions we catalogued (each is handled by a specific component):

| Corruption | Example | Handled by |
|---|---|---|
| Letter/digit typos | `Cardi0logy`, `5afe` | fuzzy string features (05) |
| Legal-suffix variants | `Private Limited` / `Pvt Ltd` / `Ltd.` | IDF-weighted blocking, fuzzy features |
| Truncation / extra words | `Brown and Warfield` vs `Brown and Warfield LLC Center` | token-set features (05) |
| Website-style names | `kimb1eolvas.com` | domain detection (02), squashed-name features (05) |
| Address reordering / abbreviation | `TX, AUSTIN, 914 39TH ST` | token features, order-insensitive keys (03) |
| House-number corruption | `914` → `91`, `0019553` → `19553` | number features, leading-zero removal (02, 05) |
| Missing address | (empty) | name-only handling (03, 07) |
| **Names in Indian scripts** | `अल टेक प्राइवेट लिमिटेड` | learned cross-script aliases (04) |
| Names/states in native script inside addresses | `…, తెలంగాణ` | same (04) |

## 1.4 The single most useful discovery: distractors are *built* to fool us

We looked at the distractors our first model wrongly accepted. They are not random lookalikes. They are
**near-duplicates of real S1 entities with one deliberate change**:

```
S1 : merna santacruz classic stardust pc   | 201 77th terrace kansas city mo
Q  : dittmer santacruz classic stardust pc | mo kansas city 201 77th ter      ← one name word swapped, address identical

S1 : vedaor corp                           | 11788 farmville rd prince edward county va
Q  : vedaor co                             | 11797 farmville rd prince edward county virginia   ← house number changed 11788→11797
```

94 % of accepted distractors land on entities that *do* have real matches, so they mostly hurt **precision**
rather than the singleton rule. String similarity alone cannot separate "a typo of the same word" from "a
different word", because both are "a bit different". What *can* separate them is looking at the **leftover
tokens** — what remains after removing the words the two records share. A typo leaves two *similar* leftovers
(`tejansh` / `tejanshyn`); a swapped word leaves two *dissimilar* ones (`merna` / `dittmer`). This insight is
the basis of the "leftover" features in [05](05_PAIR_FEATURES.md) and lifted the record-level F0.5 from 0.975 to
0.979.

## 1.5 Scripts and countries

* **US** records are entirely in Latin script.
* **India**: S1 is always Latin, but ≈ 23 % of S2 names and ≈ 13 % of S3 names are written in Indic scripts
  (Devanagari, Bengali, Tamil, Telugu, Kannada, Gujarati, Malayalam…), and ≈ 23 % of S2/S3 addresses contain a
  native-script token (usually the state). Those records often also have a *shortened* address.
* **France** (test only): Latin script with accents (`Collège Cathare`, `20 Rue de Chateaudun`).

## 1.6 What the data tells us about the labels (be aware of these)

* Some ground-truth "matches" share **only the address** with the S1 entity (e.g. a record named `Iriecto` at the
  address of `Cardiology Safe Care Inc` is labelled a match). We treat this as label noise and do not chase it.
* About 2.4 % of true matches are unreachable by our blocking and about 2.5 % are name-only records too ambiguous
  to place — so **no method can reach 100 % recall**; the practical ceiling is roughly 95–96 % record recall.

## 1.7 Consequences for the design (the short version)

1. Partition everything by `country` → smaller, faster, no recall loss. (03)
2. Because a query has at most one owner, decide *per query* who owns it (or nobody). (06, 07)
3. Blocking must tolerate reordering, truncation, typos and script changes → many independent keys. (03)
4. Learn cross-script name rewriting from the training pairs, not from a dictionary. (04)
5. Add "leftover token" features to separate typos from substitutions. (05)
6. Treat empty lists as a first-class outcome. (07)
