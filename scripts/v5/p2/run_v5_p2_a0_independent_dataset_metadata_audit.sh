#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
SCRIPT="$REPO/scripts/v5/p2/audit_v5_p2_a0_metadata.py"
OUT="$REPO/reports/v5/p2_a0_independent_dataset_metadata_audit"
LOG="$REPO/logs/v5/v5_p2_a0_independent_dataset_metadata_audit.log"

echo "===== V5 P2-A0 INDEPENDENT DATASET METADATA AUDIT ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "run_tensor_files_opened=false"
echo "run_tensor_bytes_read=false"
echo "test_tensor_contents_accessed=false"
echo "test_dataset_constructed=false"
echo

[ -e "$DATA" ] || {
    echo "STOP: dataset link is missing: $DATA"
    exit 2
}
[ -f "$SCRIPT" ] || {
    echo "STOP: audit script is missing: $SCRIPT"
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
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_a0_status=$status"

if [ -f "$OUT/V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT_COMPLETE" ]; then
    cat "$OUT/V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT_COMPLETE"
elif [ -f "$OUT/V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT_HOLD" ]; then
    cat "$OUT/V5_P2_A0_INDEPENDENT_DATASET_METADATA_AUDIT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
