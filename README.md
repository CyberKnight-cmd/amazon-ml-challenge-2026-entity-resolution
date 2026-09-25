# Amazon ML Challenge 2026 — Business Entity Resolution

Link every business in a clean reference list (Source 1) to its matching records in two noisy vendor feeds
(Source 2 and Source 3), using only names and addresses. An entity with no match must get an empty list.
Scored by per-entity macro F0.5. No external data, APIs or lookups are allowed.

## Setup
```
uv sync
uv run pytest -q
```
Data files are tab-separated and live under `data/raw/` (git-ignored).
