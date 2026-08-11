#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p0_initial24_graph"
B0="$REPO/reports/v5/p0_b0_shortcut_audit_suite"
R1="$REPO/reports/v5/p0_b0_r1_identifier_quarantine"
SCRIPT="$REPO/scripts/v5/common/run_v5_p0_b0_r1b_numeric_provenance_audit.py"
OUT="$REPO/reports/v5/p0_b0_r1b_numeric_provenance_audit"
MODEL_DIR="$REPO/models/v5/p0_b0_r1b_numeric_provenance_audit"
LOG="$REPO/logs/v5/v5_p0_b0_r1b_numeric_provenance_audit.log"

echo "===== V5 P0-B0-R1B NUMERIC PROVENANCE AUDIT ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "b0=$B0"
echo "r1=$R1"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "model_dir=$MODEL_DIR"
echo "log=$LOG"
echo

for path in "$DATA" "$B0" "$R1"; do
    if [ ! -d "$path" ]; then
        echo "STOP: required directory missing: $path"
        exit 2
    fi
done

if [ ! -f "$SCRIPT" ]; then
    echo "STOP: script missing: $SCRIPT"
    exit 2
fi
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

mkdir -p \
    "$(dirname "$OUT")" \
    "$(dirname "$MODEL_DIR")" \
    "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --root "$DATA" \
    --b0-dir "$B0" \
    --r1-dir "$R1" \
    --output-dir "$OUT" \
    --model-dir "$MODEL_DIR" \
    --window 32 \
    --stride 8 \
    --epochs 150 \
    --learning-rate 0.001 \
    --seed 47 \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_b0_r1b_status=$status"

if [ -f "$OUT/V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_PASS" ]; then
    cat "$OUT/V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_PASS"
elif [ -f "$OUT/V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_HOLD" ]; then
    cat "$OUT/V5_P0_B0_R1B_NUMERIC_PROVENANCE_AUDIT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
