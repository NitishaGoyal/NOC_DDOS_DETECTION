#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p0_initial24_graph"
A2="$REPO/reports/v5/p0_a2_feature_contract"
A2_1="$REPO/reports/v5/p0_a2_1_mask_recoverability_audit"
A3="$REPO/reports/v5/p0_a3_loader_contract_smoke"
WRAPPER="$REPO/scripts/v5/common/v5_p0_contract_loader.py"
SCRIPT="$REPO/scripts/v5/train/train_v5_p0_a4_tiny_subset_overfit.py"
OUT="$REPO/reports/v5/p0_a4_tiny_subset_overfit"
MODEL_DIR="$REPO/models/v5/p0_a4_tiny_subset_overfit"
LOG="$REPO/logs/v5/v5_p0_a4_tiny_subset_overfit.log"

echo "===== V5 P0-A4 TINY-SUBSET OVERFIT ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "a2=$A2"
echo "a2_1=$A2_1"
echo "a3=$A3"
echo "wrapper=$WRAPPER"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "model_dir=$MODEL_DIR"
echo "log=$LOG"
echo

for path in "$DATA" "$A2" "$A2_1" "$A3"; do
    if [ ! -d "$path" ]; then
        echo "STOP: required directory missing: $path"
        exit 2
    fi
done

for path in "$WRAPPER" "$SCRIPT"; do
    if [ ! -f "$path" ]; then
        echo "STOP: required script missing: $path"
        exit 2
    fi
done

if [ -e "$OUT" ]; then
    echo "STOP: output directory already exists: $OUT"
    exit 2
fi

if [ -e "$MODEL_DIR" ]; then
    echo "STOP: model directory already exists: $MODEL_DIR"
    exit 2
fi

if [ -e "$LOG" ]; then
    echo "STOP: log already exists: $LOG"
    exit 2
fi

mkdir -p "$(dirname "$OUT")" "$(dirname "$MODEL_DIR")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --root "$DATA" \
    --a2-dir "$A2" \
    --a2-1-dir "$A2_1" \
    --a3-dir "$A3" \
    --wrapper "$WRAPPER" \
    --output-dir "$OUT" \
    --model-dir "$MODEL_DIR" \
    --epochs 300 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --seed 7 \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_a4_status=$status"

if [ -f "$OUT/V5_P0_A4_TINY_SUBSET_OVERFIT_PASS" ]; then
    cat "$OUT/V5_P0_A4_TINY_SUBSET_OVERFIT_PASS"
elif [ -f "$OUT/V5_P0_A4_TINY_SUBSET_OVERFIT_HOLD" ]; then
    cat "$OUT/V5_P0_A4_TINY_SUBSET_OVERFIT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
