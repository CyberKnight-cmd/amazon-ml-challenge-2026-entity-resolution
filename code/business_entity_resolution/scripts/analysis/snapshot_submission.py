"""Freeze the current submission file with its provenance, and append a row to docs/SUBMISSION_LOG.md.

Run this immediately before uploading matching_results.tsv to the portal:
    uv run python scripts/snapshot_submission.py --label "expected-f05 v1" --params "decoder=expected mu0=0.1" --estimate 0.98
The file is copied to outputs/submissions/<n>_<label>/ (git-ignored because it is large); the log row (committed)
carries the sha256 so the exact uploaded file can always be identified.
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess

from entity_resolution.io import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--params", required=True, help="decoder and key parameters, free text")
ap.add_argument("--estimate", default="n/a", help="offline entity-level F0.5 estimate")
ap.add_argument("--notes", default="")
ap.add_argument("--src", default=str(ROOT / "outputs" / "submission" / "matching_results.tsv"))
a = ap.parse_args()

log = ROOT / "docs" / "SUBMISSION_LOG.md"
n = sum(1 for line in log.read_text().splitlines() if re.match(r"\| \d+ \|", line)) + 1
sha = hashlib.sha256(open(a.src, "rb").read()).hexdigest()
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    commit += "+dirty" if dirty else ""
except Exception:
    commit = "no-git"
slug = re.sub(r"[^a-z0-9]+", "-", a.label.lower()).strip("-")
dest = ROOT / "outputs" / "submissions" / f"{n:02d}_{slug}"
dest.mkdir(parents=True, exist_ok=True)
shutil.copy(a.src, dest / "matching_results.tsv")
now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
(dest / "meta.json").write_text(json.dumps(dict(n=n, time=now, label=a.label, params=a.params, commit=commit, sha256=sha, estimate=a.estimate, notes=a.notes), indent=2))
row = f"| {n} | {now} | {a.label} | {a.params} | {commit} | {sha[:12]} | {a.estimate} | – | – | {a.notes} |\n"
with open(log, "a") as f:
    f.write(row)
print(f"snapshot #{n} saved in {dest}\nlog row appended:\n{row}")
if commit.endswith("+dirty"):
    print("WARNING: uncommitted changes - commit first so the code version identifies this submission exactly.")
