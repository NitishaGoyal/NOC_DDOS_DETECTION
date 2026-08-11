#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
A0="$REPO/reports/v5/p2_a0_independent_dataset_metadata_audit"
A1_R2="$REPO/reports/v5/p2_a1_r2_pair_aligned_window_contract"
A2_R2="$REPO/reports/v5/p2_a2_r2_feature_normalization_mask_contract"

LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
SCRIPT="$REPO/scripts/v5/p2/audit_v5_p2_a3_loader_contract.py"
OUT="$REPO/reports/v5/p2_a3_pair_aligned_primary58_loader_contract"
LOG="$REPO/logs/v5/v5_p2_a3_pair_aligned_primary58_loader_contract.log"

echo "===== V5 P2-A3 PAIR-ALIGNED PRIMARY58 LOADER CONTRACT ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "a0=$A0"
echo "a1_r2=$A1_R2"
echo "a2_r2=$A2_R2"
echo "loader=$LOADER"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "train_tensor_contents_accessed=sample_only"
echo "validation_tensor_contents_accessed=sample_only"
echo "test_constructor_rejected=true"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo

for path in "$DATA" "$A0" "$A1_R2" "$A2_R2"; do
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
    --loader-path "$LOADER" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_a3_status=$status"

if [ -f "$OUT/V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT_COMPLETE" ]; then
    cat "$OUT/V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT_COMPLETE"
elif [ -f "$OUT/V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT_HOLD" ]; then
    cat "$OUT/V5_P2_A3_PAIR_ALIGNED_PRIMARY58_LOADER_CONTRACT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
