#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p0_initial24_graph"
A1="$REPO/reports/v5/p0_a1/tensor_handover_audit"
A2_0="$REPO/reports/v5/p0_a2_0_feature_normalization_preflight"
A2_1="$REPO/reports/v5/p0_a2_1_mask_recoverability_audit"
SCRIPT="$REPO/scripts/v5/common/freeze_v5_p0_a2_feature_normalization_contract.py"
OUT="$REPO/reports/v5/p0_a2_feature_contract"
LOG="$REPO/logs/v5/v5_p0_a2_feature_contract_freeze.log"

echo "===== V5 P0-A2 FEATURE/NORMALIZATION CONTRACT FREEZE ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "a1=$A1"
echo "a2_0=$A2_0"
echo "a2_1=$A2_1"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo

for path in "$DATA" "$A1" "$A2_0" "$A2_1"; do
    if [ ! -d "$path" ]; then
        echo "STOP: required directory missing: $path"
        exit 2
    fi
done

if [ ! -f "$SCRIPT" ]; then
    echo "STOP: contract script missing: $SCRIPT"
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
    --a2-1-dir "$A2_1" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_a2_contract_status=$status"

if [ -f "$OUT/V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS" ]; then
    cat "$OUT/V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS"
elif [ -f "$OUT/V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_HOLD" ]; then
    cat "$OUT/V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
