#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p0_initial24_graph"
A2="$REPO/reports/v5/p0_a2_feature_contract"
A2_1="$REPO/reports/v5/p0_a2_1_mask_recoverability_audit"
A3="$REPO/reports/v5/p0_a3_loader_contract_smoke"
R1C="$REPO/reports/v5/p0_b0_r1c_final_quarantine_release"
B8="$REPO/reports/v5/p0_b8_freeze_b3_conv1d_only_architecture"
C0="$REPO/reports/v5/p0_c0_final_b3_training_protocol_lock"

WRAPPER="$REPO/scripts/v5/common/v5_p0_contract_loader.py"
B3_REFERENCE="$REPO/scripts/v5/baselines/train_v5_p0_b3_conv1d_only.py"
SCRIPT="$REPO/scripts/v5/final/train_v5_p0_c1_b3_multi_seed_validation.py"

OUT="$REPO/reports/v5/p0_c1_b3_multi_seed_validation_training"
MODEL_DIR="$REPO/models/v5/p0_c1_b3_multi_seed_validation_training"
LOG="$REPO/logs/v5/v5_p0_c1_b3_multi_seed_validation_training.log"

echo "===== V5 P0-C1 B3 MULTI-SEED VALIDATION TRAINING ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "c0=$C0"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "model_dir=$MODEL_DIR"
echo "log=$LOG"
echo "seeds=107,117,127"
echo "test_split_constructed=false"
echo "test_split_accessed=false"
echo

for path in \
    "$DATA" "$A2" "$A2_1" "$A3" "$R1C" "$B8" "$C0"
do
    [ -d "$path" ] || {
        echo "STOP: missing directory: $path"
        exit 2
    }
done

for path in "$WRAPPER" "$B3_REFERENCE" "$SCRIPT"; do
    [ -f "$path" ] || {
        echo "STOP: missing file: $path"
        exit 2
    }
done

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
    --root "$DATA" \
    --a2-dir "$A2" \
    --a2-1-dir "$A2_1" \
    --a3-dir "$A3" \
    --r1c-dir "$R1C" \
    --b8-dir "$B8" \
    --c0-dir "$C0" \
    --wrapper "$WRAPPER" \
    --b3-reference-script "$B3_REFERENCE" \
    --output-dir "$OUT" \
    --model-dir "$MODEL_DIR" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_c1_status=$status"

if [ -f "$OUT/V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING_COMPLETE" ]; then
    cat "$OUT/V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING_COMPLETE"
elif [ -f "$OUT/V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING_HOLD" ]; then
    cat "$OUT/V5_P0_C1_B3_MULTI_SEED_VALIDATION_TRAINING_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
