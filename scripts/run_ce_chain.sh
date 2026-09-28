#!/usr/bin/env bash
# Tonight's cross-encoder chain, run next to run_v2_blocking.sh: waits for the stage-2 tables it needs, scores the
# hard pairs, evaluates stage 2 with the score as a feature on held-out entities, picks the best threshold from that
# evaluation, trains the final model and writes outputs/submission/matching_results_v2ce.tsv (+ candidates).
set -euo pipefail
cd "$(dirname "$0")/.."
export ER_WORK=data/work_v2 ER_BLOCKING=v2
PY=.venv/bin/python
L=logs
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/ce_progress.log; }
wait_for() { until grep -q "$2" "$1" 2>/dev/null; do sleep 20; done; }

stamp "waiting for the test stage-2 tables"
wait_for $L/early_s2_build_test.log "\[test/US\] .* columns"
stamp "cross-encoder: score test hard pairs"
$PY scripts/ce_score.py --split test --countries France India US > $L/ce_score_test.log 2>&1
stamp "waiting for the train stage-2 tables"
wait_for $L/v2blk_s2_build_train.log "\[train/India\] .* columns"
stamp "cross-encoder: score train hard pairs"
$PY scripts/ce_score.py --split train --countries US India > $L/ce_score_train.log 2>&1
stamp "stage-2 eval with cross-encoder"
$PY scripts/stage2_train.py --ce --tag _ce > $L/ce_s2_train_eval.log 2>&1
TAU=$(grep "B stage-2 r>=" $L/ce_s2_train_eval.log | sed -E 's/.*r>=([0-9.]+): f05=([0-9.]+).*/\1 \2/' | sort -k2 -g -r | head -1 | cut -d' ' -f1)
stamp "best tau on held-out entities: $TAU"
stamp "stage-2 final with cross-encoder"
$PY scripts/stage2_train.py --ce --final --tag _ce > $L/ce_s2_train_final.log 2>&1
stamp "apply (tau=$TAU)"
$PY scripts/stage2_apply.py --ce --tag _ce --tau "$TAU" --out outputs/submission/matching_results_v2ce.tsv \
    --cand_out outputs/submission/candidate_pairs_v2ce.tsv > $L/ce_s2_apply.log 2>&1
stamp "done"
