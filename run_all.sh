#!/usr/bin/env bash
# Reproduce output/matching_results.tsv and output/candidate_pairs.tsv from the organizers' TSVs in data/raw/tsv/.
#   ./run_all.sh            full run (about 4-5 h on 16 cores + one CUDA GPU)
#   SMOKE=1 ./run_all.sh    end-to-end check on a sample of records (minutes); output is valid but not competitive
set -euo pipefail
cd "$(dirname "$0")"
export ER_WORK=${ER_WORK:-data/work} ER_BLOCKING=v2
mkdir -p "$ER_WORK" output

if [ "${SMOKE:-0}" = 1 ]; then
    N=20000; ROUNDS=60; LIMIT=(--limit_queries 20000); CE=(--max_train 5000 --n_val 2000); EPOCHS=1
else
    N=250000; ROUNDS=600; LIMIT=(); CE=(--max_train 400000); EPOCHS=3
fi
step() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

step "1/10 normalise the six TSVs";            python scripts/build_interim.py
step "2/10 learn aliases from train pairs";     python scripts/fit_aliases.py
step "3/10 stage-1 matcher (v1+v2 blocking)";   python scripts/train_model.py --n "$N" --rounds "$ROUNDS"
step "4/10 score the train world";              python scripts/run_pipeline.py --split train --countries US India --chunk 200000 "${LIMIT[@]}"
step "4/10 score the test set";                 python scripts/run_pipeline.py --split test --countries France India US --chunk 200000 "${LIMIT[@]}"
step "5/10 stage-2 tables";                     python scripts/stage2_build.py --split train --countries US India
                                                python scripts/stage2_build.py --split test --countries France India US
step "6/10 fine-tune the cross-encoder";        python scripts/ce_train.py "${CE[@]}"
if [ "$EPOCHS" = 3 ]; then
    python scripts/ce_train.py "${CE[@]}" --init "$ER_WORK/ce_model" --lr 2e-5 --out ce_model_e2
    python scripts/ce_train.py "${CE[@]}" --init "$ER_WORK/ce_model_e2" --lr 1e-5 --out ce_model_e3
    rm -rf "$ER_WORK/ce_model_e1"; mv "$ER_WORK/ce_model" "$ER_WORK/ce_model_e1"; mv "$ER_WORK/ce_model_e3" "$ER_WORK/ce_model"
fi
step "7/10 cross-encoder scores (uncertain pairs)"
python scripts/ce_score.py --split train --countries US India
python scripts/ce_score.py --split test --countries France India US
step "8/10 stage-2 held-out check";             python scripts/stage2_train.py --ce --tag _lite --drop "e_*" "sib_*"
step "9/10 stage-2 final model";                python scripts/stage2_train.py --ce --final --tag _lite --drop "e_*" "sib_*"
step "10/10 write and validate the output"
python scripts/stage2_apply.py --ce --tag _lite --tau 0.85 --out output/matching_results.tsv --cand_out output/candidate_pairs.tsv
python scripts/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/tsv/test
