#!/usr/bin/env bash
set -uo pipefail

SEED="${1:-}"
case "$SEED" in
    107|117|127) ;;
    *)
        echo "Usage: $0 {107|117|127}"
        exit 2
        ;;
esac

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
B1="$REPO/reports/v5/p2_b1_training_protocol_lock"
B0_R3="$REPO/reports/v5/p2_b0_r3_corrected_nontest_label_shortcut_audit"

LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
MODEL="$REPO/src/models/v5_p2_b3_conv1d_only_count4.py"
SCRIPT="$REPO/scripts/v5/p2/train_v5_p2_b2_single_seed.py"

MODEL_DIR="$REPO/models/v5/p2_b2_multi_seed/seed_${SEED}"
REPORT_DIR="$REPO/reports/v5/p2_b2_multi_seed_training/seed_${SEED}"
LOG="$REPO/logs/v5/v5_p2_b2_seed_${SEED}.log"

echo "===== V5 P2-B2 SINGLE-SEED TRAINING ====="
echo "seed=$SEED"
echo "repo=$REPO"
echo "data=$DATA"
echo "b1=$B1"
echo "b0_r3=$B0_R3"
echo "loader=$LOADER"
echo "model=$MODEL"
echo "script=$SCRIPT"
echo "model_dir=$MODEL_DIR"
echo "report_dir=$REPORT_DIR"
echo "log=$LOG"
echo "automatic_mixed_precision=false"
echo "threshold_tuning_performed=false"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo

for path in "$DATA" "$B1" "$B0_R3"; do
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

[ ! -e "$MODEL_DIR" ] || {
    echo "STOP: model directory already exists: $MODEL_DIR"
    exit 2
}
[ ! -e "$REPORT_DIR" ] || {
    echo "STOP: report directory already exists: $REPORT_DIR"
    exit 2
}
[ ! -e "$LOG" ] || {
    echo "STOP: log already exists: $LOG"
    exit 2
}

mkdir -p \
    "$(dirname "$MODEL_DIR")" \
    "$(dirname "$REPORT_DIR")" \
    "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --root "$DATA" \
    --b1-dir "$B1" \
    --b0-r3-dir "$B0_R3" \
    --loader-path "$LOADER" \
    --model-path "$MODEL" \
    --seed "$SEED" \
    --model-dir "$MODEL_DIR" \
    --report-dir "$REPORT_DIR" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_b2_seed_${SEED}_status=$status"

if [ -f "$REPORT_DIR/V5_P2_B2_SINGLE_SEED_TRAINING_COMPLETE" ]; then
    cat "$REPORT_DIR/V5_P2_B2_SINGLE_SEED_TRAINING_COMPLETE"
elif [ -f "$REPORT_DIR/V5_P2_B2_SINGLE_SEED_TRAINING_HOLD" ]; then
    cat "$REPORT_DIR/V5_P2_B2_SINGLE_SEED_TRAINING_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
