#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

A2="$REPO/reports/v5/p0_a2_feature_contract"
A3="$REPO/reports/v5/p0_a3_loader_contract_smoke"
B0="$REPO/reports/v5/p0_b0_shortcut_audit_suite"
R1="$REPO/reports/v5/p0_b0_r1_identifier_quarantine"
R1B="$REPO/reports/v5/p0_b0_r1b_numeric_provenance_audit"
SCRIPT="$REPO/scripts/v5/common/freeze_v5_p0_b0_r1c_final_quarantine_release.py"
OUT="$REPO/reports/v5/p0_b0_r1c_final_quarantine_release"
LOG="$REPO/logs/v5/v5_p0_b0_r1c_final_quarantine_release.log"

echo "===== V5 P0-B0-R1C FINAL QUARANTINE RELEASE ====="
echo "repo=$REPO"
echo "a2=$A2"
echo "a3=$A3"
echo "b0=$B0"
echo "r1=$R1"
echo "r1b=$R1B"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo

for path in "$A2" "$A3" "$B0" "$R1" "$R1B"; do
    if [ ! -d "$path" ]; then
        echo "STOP: required directory missing: $path"
        exit 2
    fi
done

if [ ! -f "$SCRIPT" ]; then
    echo "STOP: required script missing: $SCRIPT"
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
    --r1-dir "$R1" \
    --r1b-dir "$R1B" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_b0_r1c_status=$status"

if [ -f "$OUT/V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_PASS" ]; then
    cat "$OUT/V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_PASS"
elif [ -f "$OUT/V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_HOLD" ]; then
    cat "$OUT/V5_P0_B0_R1C_FINAL_QUARANTINE_RELEASE_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
