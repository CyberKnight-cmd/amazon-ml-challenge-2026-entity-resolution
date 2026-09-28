#!/usr/bin/env bash
# Cross-encoder on ALL stage-2 pairs (hard + confident): wait for the confident-pair scores, evaluate stage 2 on
# held-out entities, and only if it beats BEAT (the best held-out F0.5 so far) train the final model, apply it at
# the best threshold and validate the file. Artefacts carry the tag _ce2.
set -euo pipefail
cd "$(dirname "$0")/.."
export ER_WORK=data/work_v2 ER_BLOCKING=v2
PY=.venv/bin/python
L=logs
BEAT=${BEAT:-0.98934}
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/ce2_progress.log; }

stamp "waiting for confident-pair scores"
until [ -f "$ER_WORK/ce_train_India_easy.parquet" ] && [ -f "$ER_WORK/ce_test_US_easy.parquet" ] && ! pgrep -f "[c]e_score.py --split" >/dev/null; do sleep 15; done
stamp "stage-2 eval with cross-encoder on all pairs"
$PY scripts/stage2_train.py --ce --ce_all --tag _ce2 > $L/ce2_s2_train_eval.log 2>&1
BEST=$(grep "B stage-2 r>=" $L/ce2_s2_train_eval.log | sed -E 's/.*r>=([0-9.]+): f05=([0-9.]+).*/\1 \2/' | sort -k2 -g -r | head -1)
TAU=${BEST% *}; F=${BEST#* }
stamp "best on held-out entities: tau=$TAU f05=$F (to beat: $BEAT)"
if awk "BEGIN{exit !($F > $BEAT)}"; then
    stamp "better -> final + apply"
    $PY scripts/stage2_train.py --ce --ce_all --final --tag _ce2 > $L/ce2_s2_train_final.log 2>&1
    $PY scripts/stage2_apply.py --ce --ce_all --tag _ce2 --tau "$TAU" --out outputs/submission/matching_results_v2ce2.tsv \
        --cand_out outputs/submission/candidate_pairs_v2ce2.tsv > $L/ce2_s2_apply.log 2>&1
    python3 scripts/validate_submission.py --matching outputs/submission/matching_results_v2ce2.tsv \
        --candidate outputs/submission/candidate_pairs_v2ce2.tsv --test-dir data/raw/tsv/test > $L/ce2_validate.log 2>&1
    stamp "done: $(tail -n 1 $L/ce2_validate.log)"
else
    stamp "not better -> stop (keep v2ce-hard)"
fi
