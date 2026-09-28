#!/usr/bin/env bash
# Stage 2 without the features that depend on how many records an entity has (entity context e_*, sibling sib_*):
# those shift on the test set (5.8 vs 4.7 records per entity, extra orphans). Cross-encoder kept. Tag _lite.
set -euo pipefail
cd "$(dirname "$0")/.."
export ER_WORK=data/work_v2 ER_BLOCKING=v2
PY=.venv/bin/python
L=logs
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/lite_progress.log; }
stamp "lite stage-2 eval"
$PY scripts/stage2_train.py --ce --tag _lite --drop "e_*" "sib_*" > $L/lite_eval.log 2>&1
BEST=$(grep "B stage-2 r>=" $L/lite_eval.log | sed -E 's/.*r>=([0-9.]+): f05=([0-9.]+).*/\1 \2/' | sort -k2 -g -r | head -1)
TAU=${BEST% *}
stamp "best: $BEST -> final"
$PY scripts/stage2_train.py --ce --final --tag _lite --drop "e_*" "sib_*" > $L/lite_final.log 2>&1
stamp "apply tau=$TAU"
$PY scripts/stage2_apply.py --ce --tag _lite --tau "$TAU" --out outputs/submission/matching_results_lite.tsv \
    --cand_out outputs/submission/candidate_pairs_lite.tsv > $L/lite_apply.log 2>&1
python3 scripts/validate_submission.py --matching outputs/submission/matching_results_lite.tsv \
    --candidate outputs/submission/candidate_pairs_lite.tsv --test-dir data/raw/tsv/test > $L/lite_validate.log 2>&1
stamp "done: $(tail -n 1 $L/lite_validate.log)"
