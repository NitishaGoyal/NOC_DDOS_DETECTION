#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

B8="$REPO/reports/v5/p0_b8_freeze_b3_conv1d_only_architecture"
C0="$REPO/reports/v5/p0_c0_final_b3_training_protocol_lock"
C1_REPORT="$REPO/reports/v5/p0_c1_b3_multi_seed_validation_training"
C1_MODEL="$REPO/models/v5/p0_c1_b3_multi_seed_validation_training"

SCRIPT="$REPO/scripts/v5/final/freeze_v5_p0_c2_checkpoint_thresholds.py"
OUT="$REPO/reports/v5/p0_c2_final_checkpoint_threshold_freeze"
MODEL_DIR="$REPO/models/v5/p0_c2_final_checkpoint_threshold_freeze"
LOG="$REPO/logs/v5/v5_p0_c2_final_checkpoint_threshold_freeze.log"

echo "===== V5 P0-C2 FINAL CHECKPOINT + THRESHOLD FREEZE ====="
echo "repo=$REPO"
echo "b8=$B8"
echo "c0=$C0"
echo "c1_report=$C1_REPORT"
echo "c1_model=$C1_MODEL"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "model_dir=$MODEL_DIR"
echo "log=$LOG"
echo "training_performed=false"
echo "dataset_constructed=false"
echo "test_split_constructed=false"
echo "test_split_accessed=false"
echo

for path in "$B8" "$C0" "$C1_REPORT" "$C1_MODEL"; do
    [ -d "$path" ] || {
        echo "STOP: missing directory: $path"
        exit 2
    }
done

[ -f "$SCRIPT" ] || {
    echo "STOP: missing script: $SCRIPT"
    exit 2
}

[ ! -e "$OUT" ] || {
    echo "STOP: output already exists: $OUT"
    exit 2
}
[ ! -e "$MODEL_DIR" ] || {
    echo "STOP: model directory already exists: $MODEL_DIR"
    exit 2
}
[ ! -e "$LOG" ] || {
    echo "STOP: log already exists: $LOG"
    exit 2
}

mkdir -p \
    "$(dirname "$OUT")" \
    "$(dirname "$MODEL_DIR")" \
    "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --b8-dir "$B8" \
    --c0-dir "$C0" \
    --c1-report-dir "$C1_REPORT" \
    --c1-model-dir "$C1_MODEL" \
    --output-dir "$OUT" \
    --model-dir "$MODEL_DIR" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_c2_status=$status"

if [ -f "$OUT/V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE_COMPLETE" ]; then
    cat "$OUT/V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE_COMPLETE"
elif [ -f "$OUT/V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE_HOLD" ]; then
    cat "$OUT/V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
