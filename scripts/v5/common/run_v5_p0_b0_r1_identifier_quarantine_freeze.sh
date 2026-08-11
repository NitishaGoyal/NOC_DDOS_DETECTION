#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

A2="$REPO/reports/v5/p0_a2_feature_contract"
A3="$REPO/reports/v5/p0_a3_loader_contract_smoke"
B0="$REPO/reports/v5/p0_b0_shortcut_audit_suite"
INVENTORY="$REPO/reports/v5/p0_b0_r1a_identifier_inventory"
SCRIPT="$REPO/scripts/v5/common/freeze_v5_p0_b0_r1_identifier_quarantine.py"
OUT="$REPO/reports/v5/p0_b0_r1_identifier_quarantine"
LOG="$REPO/logs/v5/v5_p0_b0_r1_identifier_quarantine.log"

echo "===== V5 P0-B0-R1 IDENTIFIER QUARANTINE FREEZE ====="
echo "repo=$REPO"
echo "a2=$A2"
echo "a3=$A3"
echo "b0=$B0"
echo "inventory=$INVENTORY"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo

for path in "$A2" "$A3" "$B0" "$INVENTORY"; do
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
if [ -e "$LOG" ]; then
    echo "STOP: log already exists: $LOG"
    exit 2
fi

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --a2-dir "$A2" \
    --a3-dir "$A3" \
    --b0-dir "$B0" \
    --inventory-dir "$INVENTORY" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_b0_r1_status=$status"

if [ -f "$OUT/V5_P0_B0_R1_IDENTIFIER_QUARANTINE_PASS" ]; then
    cat "$OUT/V5_P0_B0_R1_IDENTIFIER_QUARANTINE_PASS"
elif [ -f "$OUT/V5_P0_B0_R1_IDENTIFIER_QUARANTINE_HOLD" ]; then
    cat "$OUT/V5_P0_B0_R1_IDENTIFIER_QUARANTINE_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
