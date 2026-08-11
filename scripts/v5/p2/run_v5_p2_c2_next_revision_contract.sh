#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

C1="$REPO/reports/v5/p2_c1_scenario_root_cause_interpretation"
SCRIPT="$REPO/scripts/v5/p2/freeze_v5_p2_c2_next_revision_contract.py"
OUT="$REPO/reports/v5/p2_c2_next_dataset_model_revision_contract"
LOG="$REPO/logs/v5/v5_p2_c2_next_dataset_model_revision_contract.log"

echo "===== V5 P2-C2 NEXT DATASET + MODEL REVISION CONTRACT ====="
echo "repo=$REPO"
echo "c1=$C1"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "contract_source=P2_C1_only"
echo "p2_prediction_cache_accessed=false"
echo "p2_checkpoint_loaded=false"
echo "p2_test_directory_enumerated=false"
echo "p2_test_tensor_contents_accessed=false"
echo "p2_test_inference_rerun=false"
echo

[ -d "$C1" ] || {
    echo "STOP: missing C1 directory: $C1"
    exit 2
}
[ -f "$SCRIPT" ] || {
    echo "STOP: missing contract script: $SCRIPT"
    exit 2
}
[ ! -e "$OUT" ] || {
    echo "STOP: output already exists: $OUT"
    exit 2
}
[ ! -e "$LOG" ] || {
    echo "STOP: log already exists: $LOG"
    exit 2
}

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --c1-dir "$C1" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_c2_status=$status"

if [ -f "$OUT/V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT_COMPLETE" ]; then
    cat "$OUT/V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT_COMPLETE"
elif [ -f "$OUT/V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT_HOLD" ]; then
    cat "$OUT/V5_P2_C2_NEXT_DATASET_AND_MODEL_REVISION_CONTRACT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
