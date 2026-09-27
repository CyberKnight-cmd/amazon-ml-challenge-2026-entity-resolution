#!/usr/bin/env bash
# Cross-encoder variant of stage 2 on top of the v2-blocking tables in data/work_v2 (after run_v2_blocking.sh has
# built s2_train_* / s2_test_*): score the hard pairs, then evaluate / train / apply stage 2 with the score as a
# feature. Artefacts carry the tag _ce, so the plain v2 artefacts stay untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
export ER_WORK=data/work_v2 ER_BLOCKING=v2
PY=.venv/bin/python
L=logs
TAU=${TAU:-0.65}
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/ce_progress.log; }

stamp "cross-encoder: score train hard pairs"
$PY scripts/ce_score.py --split train --countries US India > $L/ce_score_train.log 2>&1
stamp "cross-encoder: score test hard pairs"
$PY scripts/ce_score.py --split test --countries France India US > $L/ce_score_test.log 2>&1
stamp "stage-2 eval with cross-encoder"
$PY scripts/stage2_train.py --ce --tag _ce > $L/ce_s2_train_eval.log 2>&1
stamp "stage-2 final with cross-encoder"
$PY scripts/stage2_train.py --ce --final --tag _ce > $L/ce_s2_train_final.log 2>&1
stamp "apply (tau=$TAU)"
$PY scripts/stage2_apply.py --ce --tag _ce --tau "$TAU" --out outputs/submission/matching_results_v2ce.tsv \
    --cand_out outputs/submission/candidate_pairs_v2ce.tsv > $L/ce_s2_apply.log 2>&1
stamp "done"
