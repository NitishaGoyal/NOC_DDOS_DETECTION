#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p0_initial24_graph"
A2="$REPO/reports/v5/p0_a2_feature_contract"
A2_1="$REPO/reports/v5/p0_a2_1_mask_recoverability_audit"
B8="$REPO/reports/v5/p0_b8_freeze_b3_conv1d_only_architecture"
C0="$REPO/reports/v5/p0_c0_final_b3_training_protocol_lock"
C1_REPORT="$REPO/reports/v5/p0_c1_b3_multi_seed_validation_training"
C1_MODEL="$REPO/models/v5/p0_c1_b3_multi_seed_validation_training"
C2_REPORT="$REPO/reports/v5/p0_c2_final_checkpoint_threshold_freeze"
C2_MODEL="$REPO/models/v5/p0_c2_final_checkpoint_threshold_freeze"

WRAPPER="$REPO/scripts/v5/common/v5_p0_contract_loader.py"
C1_SCRIPT="$REPO/scripts/v5/final/train_v5_p0_c1_b3_multi_seed_validation.py"
SCRIPT="$REPO/scripts/v5/final/evaluate_v5_p0_c3_one_shot_test.py"

OUT="$REPO/reports/v5/p0_c3_one_shot_locked_test_evaluation"
MODEL_DIR="$REPO/models/v5/p0_c3_one_shot_locked_test_evaluation"
LOG="$REPO/logs/v5/v5_p0_c3_one_shot_locked_test_evaluation.log"

echo "===== V5 P0-C3 ONE-SHOT LOCKED TEST EVALUATION ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "c2_report=$C2_REPORT"
echo "c2_model=$C2_MODEL"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "model_dir=$MODEL_DIR"
echo "log=$LOG"
echo
echo "WARNING: this is the first and only authorized P0 test evaluation."
echo "No training, checkpoint selection, or threshold calibration will occur."
echo

for path in \
    "$DATA" "$A2" "$A2_1" "$B8" "$C0" \
    "$C1_REPORT" "$C1_MODEL" "$C2_REPORT" "$C2_MODEL"
do
    [ -d "$path" ] || {
        echo "STOP: missing directory: $path"
        exit 2
    }
done

for path in "$WRAPPER" "$C1_SCRIPT" "$SCRIPT"; do
    [ -f "$path" ] || {
        echo "STOP: missing file: $path"
        exit 2
    }
done

[ ! -e "$OUT" ] || {
    echo "STOP: one-shot output already exists: $OUT"
    exit 2
}
[ ! -e "$MODEL_DIR" ] || {
    echo "STOP: one-shot model directory already exists: $MODEL_DIR"
    exit 2
}
[ ! -e "$LOG" ] || {
    echo "STOP: one-shot log already exists: $LOG"
    exit 2
}

mkdir -p \
    "$(dirname "$OUT")" \
    "$(dirname "$MODEL_DIR")" \
    "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --root "$DATA" \
    --a2-dir "$A2" \
    --a2-1-dir "$A2_1" \
    --b8-dir "$B8" \
    --c0-dir "$C0" \
    --c1-report-dir "$C1_REPORT" \
    --c1-model-dir "$C1_MODEL" \
    --c2-report-dir "$C2_REPORT" \
    --c2-model-dir "$C2_MODEL" \
    --wrapper "$WRAPPER" \
    --c1-script "$C1_SCRIPT" \
    --output-dir "$OUT" \
    --model-dir "$MODEL_DIR" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_c3_status=$status"

if [ -f "$OUT/V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION_COMPLETE" ]; then
    cat "$OUT/V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION_COMPLETE"
elif [ -f "$OUT/V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION_HOLD" ]; then
    cat "$OUT/V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
