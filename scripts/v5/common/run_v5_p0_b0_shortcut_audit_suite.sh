#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p0_initial24_graph"
A2="$REPO/reports/v5/p0_a2_feature_contract"
A2_1="$REPO/reports/v5/p0_a2_1_mask_recoverability_audit"
A3="$REPO/reports/v5/p0_a3_loader_contract_smoke"
A4="$REPO/reports/v5/p0_a4_tiny_subset_overfit"
WRAPPER="$REPO/scripts/v5/common/v5_p0_contract_loader.py"
SCRIPT="$REPO/scripts/v5/common/run_v5_p0_b0_shortcut_audit_suite.py"
OUT="$REPO/reports/v5/p0_b0_shortcut_audit_suite"
MODEL_DIR="$REPO/models/v5/p0_b0_shortcut_audit_suite"
LOG="$REPO/logs/v5/v5_p0_b0_shortcut_audit_suite.log"

echo "===== V5 P0-B0 SHORTCUT AUDIT SUITE ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "a2=$A2"
echo "a2_1=$A2_1"
echo "a3=$A3"
echo "a4=$A4"
echo "wrapper=$WRAPPER"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "model_dir=$MODEL_DIR"
echo "log=$LOG"
echo

for path in "$DATA" "$A2" "$A2_1" "$A3" "$A4"; do
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
    --a4-dir "$A4" \
    --wrapper "$WRAPPER" \
    --output-dir "$OUT" \
    --model-dir "$MODEL_DIR" \
    --probe-epochs 200 \
    --traffic-epochs 30 \
    --learning-rate 0.001 \
    --seed 11 \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_b0_status=$status"

if [ -f "$OUT/V5_P0_B0_SHORTCUT_AUDIT_SUITE_PASS" ]; then
    cat "$OUT/V5_P0_B0_SHORTCUT_AUDIT_SUITE_PASS"
elif [ -f "$OUT/V5_P0_B0_SHORTCUT_AUDIT_SUITE_HOLD" ]; then
    cat "$OUT/V5_P0_B0_SHORTCUT_AUDIT_SUITE_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
