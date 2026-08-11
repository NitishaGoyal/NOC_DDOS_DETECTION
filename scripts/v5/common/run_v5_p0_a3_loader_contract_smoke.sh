#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p0_initial24_graph"
A2="$REPO/reports/v5/p0_a2_feature_contract"
A2_1="$REPO/reports/v5/p0_a2_1_mask_recoverability_audit"
A3_0="$REPO/reports/v5/p0_a3_0_loader_interface"
WRAPPER="$REPO/scripts/v5/common/v5_p0_contract_loader.py"
SCRIPT="$REPO/scripts/v5/common/audit_v5_p0_a3_loader_contract_smoke.py"
OUT="$REPO/reports/v5/p0_a3_loader_contract_smoke"
LOG="$REPO/logs/v5/v5_p0_a3_loader_contract_smoke.log"

echo "===== V5 P0-A3 LOADER CONTRACT AND SMOKE ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "a2=$A2"
echo "a2_1=$A2_1"
echo "a3_0=$A3_0"
echo "wrapper=$WRAPPER"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo

for path in "$DATA" "$A2" "$A2_1" "$A3_0"; do
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

if [ -e "$LOG" ]; then
    echo "STOP: log already exists: $LOG"
    exit 2
fi

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --root "$DATA" \
    --a2-dir "$A2" \
    --a2-1-dir "$A2_1" \
    --a3-0-dir "$A3_0" \
    --wrapper "$WRAPPER" \
    --output-dir "$OUT" \
    --window 32 \
    --stride 8 \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_a3_status=$status"

if [ -f "$OUT/V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS" ]; then
    cat "$OUT/V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_PASS"
elif [ -f "$OUT/V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_HOLD" ]; then
    cat "$OUT/V5_P0_A3_LOADER_CONTRACT_AND_SMOKE_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
