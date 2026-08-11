#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
A0="$REPO/reports/v5/p2_a0_independent_dataset_metadata_audit"
A1_R1="$REPO/reports/v5/p2_a1_r1_pair_length_window_shortcut_audit"
A1_R2="$REPO/reports/v5/p2_a1_r2_pair_aligned_window_contract"

SCRIPT="$REPO/scripts/v5/p2/freeze_v5_p2_a2_feature_normalization_mask_contract.py"
OUT="$REPO/reports/v5/p2_a2_feature_normalization_mask_contract"
LOG="$REPO/logs/v5/v5_p2_a2_feature_normalization_mask_contract.log"

echo "===== V5 P2-A2 FEATURE/NORMALIZATION/MASK CONTRACT ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "a0=$A0"
echo "a1_r1=$A1_R1"
echo "a1_r2=$A1_R2"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "train_tensor_contents_accessed=sample_only"
echo "validation_tensor_contents_accessed=sample_only"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo

for path in "$DATA" "$A0" "$A1_R1" "$A1_R2"; do
    [ -d "$path" ] || {
        echo "STOP: missing directory: $path"
        exit 2
    }
done

[ -f "$SCRIPT" ] || {
    echo "STOP: missing script: $SCRIPT"
    exit 2
}

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
    --a0-dir "$A0" \
    --a1-r1-dir "$A1_R1" \
    --a1-r2-dir "$A1_R2" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_a2_status=$status"

if [ -f "$OUT/V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT_COMPLETE" ]; then
    cat "$OUT/V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT_COMPLETE"
elif [ -f "$OUT/V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT_HOLD" ]; then
    cat "$OUT/V5_P2_A2_FEATURE_NORMALIZATION_AND_MASK_CONTRACT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
