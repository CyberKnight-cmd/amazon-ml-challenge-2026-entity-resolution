#!/usr/bin/env bash
# Stage 2 from per-pair evidence only: no stage-1 p, no p-derived query context, no entity context, no sibling and
# no S1-duplicate counts (all shift with S1 size / population on test). Cross-encoder on all pairs. Tag _pf.
set -euo pipefail
cd "$(dirname "$0")/.."
export ER_WORK=data/work_v2 ER_BLOCKING=v2
PY=.venv/bin/python
L=logs
DROP="p q_rank q_ncand q_psum q_pother e_* sib_* s1_name_dup s1_addr_dup"
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/pf_progress.log; }
stamp "p-free stage-2 eval"
$PY scripts/stage2_train.py --ce --ce_all --tag _pf --rounds 1200 --drop $DROP > $L/pf_eval.log 2>&1
BEST=$(grep "B stage-2 r>=" $L/pf_eval.log | sed -E 's/.*r>=([0-9.]+): f05=([0-9.]+).*/\1 \2/' | sort -k2 -g -r | head -1)
TAU=${BEST% *}
stamp "best: $BEST -> final"
$PY scripts/stage2_train.py --ce --ce_all --final --tag _pf --drop $DROP > $L/pf_final.log 2>&1
stamp "apply tau=$TAU"
$PY scripts/stage2_apply.py --ce --ce_all --tag _pf --tau "$TAU" --out outputs/submission/matching_results_pf.tsv \
    --cand_out outputs/submission/candidate_pairs_pf.tsv > $L/pf_apply.log 2>&1
python3 scripts/validate_submission.py --matching outputs/submission/matching_results_pf.tsv \
    --candidate outputs/submission/candidate_pairs_pf.tsv --test-dir data/raw/tsv/test > $L/pf_validate.log 2>&1
stamp "done: $(tail -n 1 $L/pf_validate.log)"
