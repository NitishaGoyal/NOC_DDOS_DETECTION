#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
A1_R2="$REPO/reports/v5/p2_a1_r2_pair_aligned_window_contract"
A2_R2="$REPO/reports/v5/p2_a2_r2_feature_normalization_mask_contract"
A3="$REPO/reports/v5/p2_a3_pair_aligned_primary58_loader_contract"

LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
MODEL="$REPO/src/models/v5_frozen_b3_conv1d_only.py"
SCRIPT="$REPO/scripts/v5/p2/run_v5_p2_a4_loader_integration_tiny_overfit.py"

OUT="$REPO/reports/v5/p2_a4_loader_integration_tiny_overfit"
LOG="$REPO/logs/v5/v5_p2_a4_loader_integration_tiny_overfit.log"

echo "===== V5 P2-A4 LOADER INTEGRATION + TINY OVERFIT ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "a1_r2=$A1_R2"
echo "a2_r2=$A2_R2"
echo "a3=$A3"
echo "loader=$LOADER"
echo "model=$MODEL"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "training_scope=fixed_train_only_audit_batch"
echo "audit_weights_saved=false"
echo "validation_tensor_contents_accessed=false"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo

for path in "$DATA" "$A1_R2" "$A2_R2" "$A3"; do
    [ -d "$path" ] || {
        echo "STOP: missing directory: $path"
        exit 2
    }
done

for path in "$LOADER" "$MODEL" "$SCRIPT"; do
    [ -f "$path" ] || {
        echo "STOP: missing file: $path"
        exit 2
    }
done

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
    --root "$DATA" \
    --a1-r2-dir "$A1_R2" \
    --a2-r2-dir "$A2_R2" \
    --a3-dir "$A3" \
    --loader-path "$LOADER" \
    --model-path "$MODEL" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_a4_status=$status"

if [ -f "$OUT/V5_P2_A4_LOADER_INTEGRATION_AND_TINY_OVERFIT_COMPLETE" ]; then
    cat "$OUT/V5_P2_A4_LOADER_INTEGRATION_AND_TINY_OVERFIT_COMPLETE"
elif [ -f "$OUT/V5_P2_A4_LOADER_INTEGRATION_AND_TINY_OVERFIT_HOLD" ]; then
    cat "$OUT/V5_P2_A4_LOADER_INTEGRATION_AND_TINY_OVERFIT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
