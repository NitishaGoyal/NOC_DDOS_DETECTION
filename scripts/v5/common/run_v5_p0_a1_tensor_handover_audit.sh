#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p0_initial24_graph"
SCRIPT="$REPO/scripts/v5/common/audit_v5_p0_tensor_handover.py"
OUT="$REPO/reports/v5/p0_a1/tensor_handover_audit"
LOG="$REPO/logs/v5/v5_p0_a1_tensor_handover_audit.log"

echo "===== V5 P0-A1 TENSOR HANDOVER AUDIT ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo

if [ ! -d "$DATA" ]; then
    echo "STOP: dataset root missing: $DATA"
    exit 2
fi

if [ ! -f "$SCRIPT" ]; then
    echo "STOP: audit script missing: $SCRIPT"
    exit 2
fi

if [ -e "$OUT" ]; then
    echo "STOP: output directory already exists: $OUT"
    exit 2
fi

if [ -e "$LOG" ]; then
    echo "STOP: log already exists: $LOG"
    exit 2
fi

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --root "$DATA" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_a1_status=$status"

if [ -f "$OUT/V5_P0_A1_TENSOR_HANDOVER_AUDIT_PASS" ]; then
    cat "$OUT/V5_P0_A1_TENSOR_HANDOVER_AUDIT_PASS"
elif [ -f "$OUT/V5_P0_A1_TENSOR_HANDOVER_AUDIT_HOLD" ]; then
    cat "$OUT/V5_P0_A1_TENSOR_HANDOVER_AUDIT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
