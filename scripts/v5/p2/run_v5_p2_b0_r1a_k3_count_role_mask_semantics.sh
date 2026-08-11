#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
A1_R2="$REPO/reports/v5/p2_a1_r2_pair_aligned_window_contract"
A3="$REPO/reports/v5/p2_a3_pair_aligned_primary58_loader_contract"
A4="$REPO/reports/v5/p2_a4_loader_integration_tiny_overfit"
B0="$REPO/reports/v5/p2_b0_nontest_label_shortcut_audit"
B0_R1="$REPO/reports/v5/p2_b0_r1_k3_count_semantics_head_compatibility"

MODEL="$REPO/src/models/v5_frozen_b3_conv1d_only.py"
SCRIPT="$REPO/scripts/v5/p2/audit_v5_p2_b0_r1a_k3_count_role_mask_semantics.py"

OUT="$REPO/reports/v5/p2_b0_r1a_k3_count_role_mask_semantics_audit"
LOG="$REPO/logs/v5/v5_p2_b0_r1a_k3_count_role_mask_semantics_audit.log"

echo "===== V5 P2-B0-R1A K3 COUNT + ROLE-MASK SEMANTICS ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "a1_r2=$A1_R2"
echo "a3=$A3"
echo "a4=$A4"
echo "historical_b0_hold=$B0"
echo "historical_b0_r1_hold=$B0_R1"
echo "model=$MODEL"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "train_tensor_contents_accessed=true"
echo "validation_tensor_contents_accessed=true"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo

for path in \
    "$DATA" \
    "$A1_R2" \
    "$A3" \
    "$A4" \
    "$B0" \
    "$B0_R1"
do
    [ -d "$path" ] || {
        echo "STOP: missing directory: $path"
        exit 2
    }
done

for path in "$MODEL" "$SCRIPT"; do
    [ -f "$path" ] || {
        echo "STOP: missing file: $path"
        exit 2
    }
done

[ ! -e "$OUT" ] || {
    echo "STOP: output already exists: $OUT"
    exit 2
}
[ ! -e "$LOG" ] || {
    echo "STOP: log already exists: $LOG"
    exit 2
}

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --root "$DATA" \
    --a1-r2-dir "$A1_R2" \
    --a3-dir "$A3" \
    --a4-dir "$A4" \
    --b0-dir "$B0" \
    --b0-r1-dir "$B0_R1" \
    --model-path "$MODEL" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_b0_r1a_status=$status"

if [ -f "$OUT/V5_P2_B0_R1A_K3_COUNT_AND_ROLE_MASK_SEMANTICS_AUDIT_COMPLETE" ]; then
    cat "$OUT/V5_P2_B0_R1A_K3_COUNT_AND_ROLE_MASK_SEMANTICS_AUDIT_COMPLETE"
elif [ -f "$OUT/V5_P2_B0_R1A_K3_COUNT_AND_ROLE_MASK_SEMANTICS_AUDIT_HOLD" ]; then
    cat "$OUT/V5_P2_B0_R1A_K3_COUNT_AND_ROLE_MASK_SEMANTICS_AUDIT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
