#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p0_initial24_graph"
A1="$REPO/reports/v5/p0_a1/tensor_handover_audit"
A2_0="$REPO/reports/v5/p0_a2_0_feature_normalization_preflight"
SCRIPT="$REPO/scripts/v5/common/audit_v5_p0_a2_1_mask_recoverability.py"
OUT="$REPO/reports/v5/p0_a2_1_mask_recoverability_audit"
LOG="$REPO/logs/v5/v5_p0_a2_1_mask_recoverability_audit.log"

echo "===== V5 P0-A2-1 MASK RECOVERABILITY AUDIT ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "a1=$A1"
echo "a2_0=$A2_0"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo

for path in "$DATA" "$A1" "$A2_0"; do
    if [ ! -d "$path" ]; then
        echo "STOP: required directory missing: $path"
        exit 2
    fi
done

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
    --a1-dir "$A1" \
    --a2-0-dir "$A2_0" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_a2_1_status=$status"

if [ -f "$OUT/V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_PASS" ]; then
    cat "$OUT/V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_PASS"
elif [ -f "$OUT/V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_HOLD" ]; then
    cat "$OUT/V5_P0_A2_1_MASK_RECOVERABILITY_AUDIT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
