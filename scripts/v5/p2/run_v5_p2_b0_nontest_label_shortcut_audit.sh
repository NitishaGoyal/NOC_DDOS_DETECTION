#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
A0="$REPO/reports/v5/p2_a0_independent_dataset_metadata_audit"
A1_R2="$REPO/reports/v5/p2_a1_r2_pair_aligned_window_contract"
A2_R2="$REPO/reports/v5/p2_a2_r2_feature_normalization_mask_contract"
A3="$REPO/reports/v5/p2_a3_pair_aligned_primary58_loader_contract"
A4="$REPO/reports/v5/p2_a4_loader_integration_tiny_overfit"

LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
SCRIPT="$REPO/scripts/v5/p2/audit_v5_p2_b0_nontest_label_shortcuts.py"

OUT="$REPO/reports/v5/p2_b0_nontest_label_shortcut_audit"
LOG="$REPO/logs/v5/v5_p2_b0_nontest_label_shortcut_audit.log"

echo "===== V5 P2-B0 NON-TEST LABEL + SHORTCUT AUDIT ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "a0=$A0"
echo "a1_r2=$A1_R2"
echo "a2_r2=$A2_R2"
echo "a3=$A3"
echo "a4=$A4"
echo "loader=$LOADER"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "train_tensor_contents_accessed=true"
echo "validation_tensor_contents_accessed=true"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo

for path in "$DATA" "$A0" "$A1_R2" "$A2_R2" "$A3" "$A4"; do
    [ -d "$path" ] || {
        echo "STOP: missing directory: $path"
        exit 2
    }
done

for path in "$LOADER" "$SCRIPT"; do
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
    --a0-dir "$A0" \
    --a1-r2-dir "$A1_R2" \
    --a2-r2-dir "$A2_R2" \
    --a3-dir "$A3" \
    --a4-dir "$A4" \
    --loader-path "$LOADER" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_b0_status=$status"

if [ -f "$OUT/V5_P2_B0_NONTEST_LABEL_AND_SHORTCUT_AUDIT_COMPLETE" ]; then
    cat "$OUT/V5_P2_B0_NONTEST_LABEL_AND_SHORTCUT_AUDIT_COMPLETE"
elif [ -f "$OUT/V5_P2_B0_NONTEST_LABEL_AND_SHORTCUT_AUDIT_HOLD" ]; then
    cat "$OUT/V5_P2_B0_NONTEST_LABEL_AND_SHORTCUT_AUDIT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
