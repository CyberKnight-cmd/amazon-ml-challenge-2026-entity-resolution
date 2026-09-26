"""Stand-in for the organizers' utils/validate_submission.py (stdlib only): checks every rule stated on the
submission page for matching_results.tsv and candidate_pairs.tsv. Run the official script too when available.

    python3 scripts/validate_submission.py --matching M.tsv --candidate C.tsv --test-dir data/raw/tsv/test
"""
import argparse
import sys
from pathlib import Path


def ids(path: Path) -> list[str]:
    """entity_id column of a raw test source file."""
    with open(path, encoding="utf-8") as f:
        next(f)
        return [line.split("\t", 1)[0] for line in f]


def check(path: Path, header: tuple[str, str], s1: list[str], others: set[str], issues: list[str]) -> dict[str, list[str]]:
    """Validate one output file; returns its lists keyed by S1 id."""
    rows: dict[str, list[str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        head = f.readline().rstrip("\n").split("\t")
        if tuple(head) != header:
            issues.append(f"{path.name}: header {head} != {list(header)}")
        for n, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                issues.append(f"{path.name}:{n}: expected 2 tab-separated columns, got {len(parts)}")
                continue
            k, v = parts
            if k in rows:
                issues.append(f"{path.name}:{n}: duplicate source1_entity_id {k}")
            lst = [x for x in v.split(",")] if v else []
            if '"' in v:
                issues.append(f"{path.name}:{n}: quote character in the id list")
            if len(set(lst)) != len(lst):
                issues.append(f"{path.name}:{n}: duplicate ids within the list of {k}")
            bad = [x for x in lst if x not in others]
            if bad:
                issues.append(f"{path.name}:{n}: {len(bad)} ids are not S2/S3 test ids (e.g. {bad[0]!r})")
            rows[k] = lst
            if len(issues) > 50:
                return rows
    missing = set(s1) - set(rows)
    extra = set(rows) - set(s1)
    if missing:
        issues.append(f"{path.name}: {len(missing):,} test S1 entities missing")
    if extra:
        issues.append(f"{path.name}: {len(extra):,} rows are not test S1 entities")
    return rows


ap = argparse.ArgumentParser()
ap.add_argument("--matching", required=True)
ap.add_argument("--candidate", default=None)
ap.add_argument("--test-dir", default="data/raw/tsv/test")
a = ap.parse_args()
td = Path(a.test_dir)
s1 = ids(td / "test_source1.tsv")
others = set(ids(td / "test_source2.tsv")) | set(ids(td / "test_source3.tsv"))
issues: list[str] = []
m = check(Path(a.matching), ("source1_entity_id", "matched_entity_ids"), s1, others, issues)
used = [x for v in m.values() for x in v]
if len(used) != len(set(used)):
    print(f"note: {len(used) - len(set(used)):,} S2/S3 ids appear under more than one S1 entity (allowed, but unusual)")
if a.candidate:
    c = check(Path(a.candidate), ("source1_entity_id", "candidate_entity_ids"), s1, others, issues)
    notcand = sum(1 for k, v in m.items() for x in v if x not in set(c.get(k, ())))
    if notcand:
        issues.append(f"{notcand:,} matched ids are not in the candidate list of their S1 entity")
    n = sum(len(v) for v in c.values())
    print(f"candidates: {n:,} pairs, {n / len(s1):.2f} per S1 entity")
print(f"matches: {len(used):,} ids, {sum(1 for v in m.values() if not v) / len(s1):.1%} empty lists")
if issues:
    print("FAIL")
    for i, x in enumerate(issues, 1):
        print(f"{i}. {x}")
    sys.exit(1)
print("PASS")
