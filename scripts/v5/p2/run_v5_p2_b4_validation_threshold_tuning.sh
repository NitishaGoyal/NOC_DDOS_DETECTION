#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
A1_R2="$REPO/reports/v5/p2_a1_r2_pair_aligned_window_contract"
B1="$REPO/reports/v5/p2_b1_training_protocol_lock"
B3="$REPO/reports/v5/p2_b3_seed_checkpoint_selection"

LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
MODEL_SOURCE="$REPO/src/models/v5_p2_b3_conv1d_only_count4.py"
SCRIPT="$REPO/scripts/v5/p2/tune_v5_p2_b4_validation_thresholds.py"

OUT="$REPO/reports/v5/p2_b4_validation_threshold_tuning"
LOG="$REPO/logs/v5/v5_p2_b4_validation_threshold_tuning.log"

echo "===== V5 P2-B4 VALIDATION THRESHOLD TUNING ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "a1_r2=$A1_R2"
echo "b1=$B1"
echo "b3=$B3"
echo "loader=$LOADER"
echo "model_source=$MODEL_SOURCE"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "selected_seed=127"
echo "selected_epoch=59"
echo "threshold_tuning_split=validation"
echo "training_performed=false"
echo "model_weights_changed=false"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo "test_evaluation_authorized=false"
echo

for path in "$DATA" "$A1_R2" "$B1" "$B3"; do
    [ -d "$path" ] || {
        echo "STOP: missing directory: $path"
        exit 2
    }
done

for path in "$LOADER" "$MODEL_SOURCE" "$SCRIPT"; do
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
    --b1-dir "$B1" \
    --b3-dir "$B3" \
    --loader-path "$LOADER" \
    --model-source-path "$MODEL_SOURCE" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_b4_status=$status"

if [ -f "$OUT/V5_P2_B4_VALIDATION_THRESHOLD_TUNING_COMPLETE" ]; then
    cat "$OUT/V5_P2_B4_VALIDATION_THRESHOLD_TUNING_COMPLETE"
elif [ -f "$OUT/V5_P2_B4_VALIDATION_THRESHOLD_TUNING_HOLD" ]; then
    cat "$OUT/V5_P2_B4_VALIDATION_THRESHOLD_TUNING_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
