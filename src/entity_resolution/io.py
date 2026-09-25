"""Loading raw TSVs and caching normalized records as Parquet in data/interim."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from .normalize import norm_addr, norm_name

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "tsv"
INTERIM = ROOT / "data" / "interim"
# where derived artifacts (pair tables, models, predictions) are written; override with ER_WORK
WORK = Path(os.environ.get("ER_WORK", INTERIM))


def raw_path(split: str, source: int | str) -> Path:
    """Path of a raw TSV, e.g. raw_path('train', 2) or raw_path('train', 'ground_truth')."""
    name = f"{split}_source{source}" if source != "ground_truth" else f"{split}_ground_truth"
    return RAW / split / f"{name}.tsv"


def read_tsv(path: Path) -> pl.DataFrame:
    """All columns as strings; empty fields become ''. Quotes are data, not CSV quoting."""
    return pl.read_csv(path, separator="\t", infer_schema=False, quote_char=None).fill_null("")


def _norm_chunk(args: tuple[list[str], list[str]]):
    """Worker: normalize one chunk of (names, addresses); returns cleaned names, domain flags, cleaned addresses."""
    names, addrs = args
    nn, dom = zip(*(norm_name(x) for x in names)) if names else ((), ())
    an = [norm_addr(x) for x in addrs]
    return list(nn), list(dom), an


def normalize_frame(df: pl.DataFrame, workers: int = 12, chunk: int = 250_000) -> pl.DataFrame:
    """Normalize a raw record table in parallel; returns columns id, country, name_n, addr_n, is_domain, name_sq."""
    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    jobs = [(names[i : i + chunk], addrs[i : i + chunk]) for i in range(0, len(names), chunk)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_norm_chunk, jobs))
    name_n = [x for r in res for x in r[0]]
    is_dom = [x for r in res for x in r[1]]
    addr_n = [x for r in res for x in r[2]]
    return df.select(
        pl.col("entity_id").alias("id"),
        pl.col("country").cast(pl.Categorical),
        pl.Series("name_n", name_n),
        pl.Series("addr_n", addr_n),
        pl.Series("is_domain", is_dom),
    ).with_columns(pl.col("name_n").str.replace_all(" ", "", literal=True).alias("name_sq"))


def load_normalized(split: str, source: int, refresh: bool = False) -> pl.DataFrame:
    """Normalized records for one split/source; built once from the TSV then cached."""
    cache = INTERIM / f"{split}_s{source}.parquet"
    if cache.exists() and not refresh:
        return pl.read_parquet(cache)
    df = normalize_frame(read_tsv(raw_path(split, source)))
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache, compression="zstd")
    return df


def load_truth(split: str = "train") -> pl.DataFrame:
    """Long-format ground truth: one row per (s1, matched id)."""
    gt = read_tsv(raw_path(split, "ground_truth"))
    return (
        gt.filter(pl.col("matched_entity_ids") != "")
        .select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("m"))
        .explode("m")
        .rename({"source1_entity_id": "s1", "m": "other"})
    )
