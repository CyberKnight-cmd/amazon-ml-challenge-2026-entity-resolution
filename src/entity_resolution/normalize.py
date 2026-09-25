"""Text normalization for business names and addresses.

Deliberately language-agnostic: no country-specific rules or hard-coded lookup tables, because the test
set contains a country (France) that never appears in training. Token aliases (st/street, tx/texas, ...)
are *learned* from the training pairs elsewhere (see aliases.py), not listed here.
"""

from __future__ import annotations

import re
import unicodedata

# Indic blocks U+0900-U+0DFF are kept (letters *and* combining vowel signs, which \w would drop).
_KEEP = re.compile(r"[^0-9a-zऀ-෿\s]")
_WS = re.compile(r"\s+")
_COMBINING = re.compile(r"[̀-ͯ]")
_ZERO_WIDTH = str.maketrans("", "", "​‌‍⁠﻿")
_LATIN_EXTRA = str.maketrans(
    {"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "đ": "d", "ł": "l", "ı": "i", "þ": "th", "ð": "d"}
)
_DOMAIN = re.compile(r"^[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.(?:com|net|org|in|fr|co|biz|info|io|us)$")
_DOMAIN_TLD = re.compile(r"\.(?:co\.in|co\.uk|com|net|org|in|fr|co|biz|info|io|us)$")
# components that mean "no value" in the address field
_MISSING = frozenset({"", "n/a", "na", "nan", "null", "none", "nil", "-", "--"})


def fold(s: str) -> str:
    """Lowercase and strip Latin diacritics; Indic scripts pass through untouched."""
    s = s.lower()
    if not s.isascii():
        s = unicodedata.normalize("NFKD", s).translate(_ZERO_WIDTH)
        s = _COMBINING.sub("", s).translate(_LATIN_EXTRA)
    return s


def _tokens(s: str) -> list[str]:
    """Split text into tokens after '&' -> 'and' and replacing punctuation by spaces (Indic letters and vowel signs are kept)."""
    s = s.replace("&", " and ")
    return _WS.sub(" ", _KEEP.sub(" ", s)).strip().split(" ") if s.strip() else []


def _dedupe_runs(tokens: list[str]) -> list[str]:
    """Collapse immediately repeated tokens ('heritage heritage' -> 'heritage')."""
    out: list[str] = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def norm_name(raw: str) -> tuple[str, bool]:
    """Return (normalized name, is_domain_style). Domain-style names ("acme-robotics.com") become one token."""
    s = fold(raw).strip()
    if " " not in s and _DOMAIN.match(s):
        stem = _DOMAIN_TLD.sub("", s).replace("-", "").replace(".", "")
        return stem, True
    return " ".join(_dedupe_runs(_tokens(s))), False


def _strip_zeros(t: str) -> str:
    """Remove leading zeros from an all-digit token ('0019553' -> '19553'); other tokens unchanged."""
    return (t.lstrip("0") or "0") if t.isdigit() else t


def norm_addr(raw: str) -> str:
    """Normalize an address; 'missing' components (N/A, NULL, nan) are dropped, leading zeros of numbers removed."""
    parts = []
    for comp in fold(raw).split(","):
        if comp.strip() in _MISSING:
            continue
        parts.extend(_strip_zeros(t) for t in _tokens(comp))
    return " ".join(parts)


def squash(s: str) -> str:
    """Remove spaces: lets 'kimble olva' compare against a domain-style 'kimbleolva'."""
    return s.replace(" ", "")
