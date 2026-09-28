#!/usr/bin/env bash
# Rebuild every model stage with v2 blocking (v1 candidates + generator-aware keys) in data/work_v2.
# Stage-2 evaluation (held-out entities, same protocol as stage2-v1) is the go/no-go number.
set -euo pipefail
cd "$(dirname "$0")/.."
export ER_WORK=data/work_v2 ER_BLOCKING=v2
PY=.venv/bin/python
L=logs
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/v2blk_progress.log; }

stamp "train stage-1 matcher"
$PY scripts/train_model.py --n 250000 --rounds 600 > $L/v2blk_train_model.log 2>&1
stamp "score train world and test set (parallel)"
$PY scripts/run_pipeline.py --split train --countries US India --chunk 200000 > $L/v2blk_score_train.log 2>&1 &
P1=$!
$PY scripts/run_pipeline.py --split test --countries France India US --chunk 200000 > $L/v2blk_score_test.log 2>&1 &
P2=$!
wait $P1; stamp "train world scored"
wait $P2; stamp "test set scored"
stamp "stage-2 build"
$PY scripts/stage2_build.py --split train --countries US India > $L/v2blk_s2_build_train.log 2>&1
$PY scripts/stage2_build.py --split test --countries France India US > $L/v2blk_s2_build_test.log 2>&1
stamp "stage-2 eval (held-out entities)"
$PY scripts/stage2_train.py > $L/v2blk_s2_train_eval.log 2>&1
stamp "stage-2 final"
$PY scripts/stage2_train.py --final > $L/v2blk_s2_train_final.log 2>&1
stamp "apply to test"
$PY scripts/stage2_apply.py --tau 0.65 --out outputs/submission/matching_results_v2blk.tsv \
    --cand_out outputs/submission/candidate_pairs_v2blk.tsv > $L/v2blk_s2_apply.log 2>&1
stamp "done"
