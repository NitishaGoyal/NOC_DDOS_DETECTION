#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
A0="$REPO/reports/v5/p2_a0_independent_dataset_metadata_audit"
A1="$REPO/reports/v5/p2_a1_tensor_schema_nontest_sample_audit"

SCRIPT="$REPO/scripts/v5/p2/audit_v5_p2_a1_r1_pair_length_window_shortcut.py"
OUT="$REPO/reports/v5/p2_a1_r1_pair_length_window_shortcut_audit"
LOG="$REPO/logs/v5/v5_p2_a1_r1_pair_length_window_shortcut_audit.log"

echo "===== V5 P2-A1-R1 PAIR-LENGTH + WINDOW-SHORTCUT AUDIT ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "a0=$A0"
echo "a1_hold=$A1"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "window=32"
echo "stride=8"
echo "train_tensor_contents_accessed=true"
echo "validation_tensor_contents_accessed=true"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo

for path in "$DATA" "$A0" "$A1"; do
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
    --a1-dir "$A1" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_a1_r1_status=$status"

if [ -f "$OUT/V5_P2_A1_R1_PAIR_LENGTH_AND_WINDOW_SHORTCUT_AUDIT_COMPLETE" ]; then
    cat "$OUT/V5_P2_A1_R1_PAIR_LENGTH_AND_WINDOW_SHORTCUT_AUDIT_COMPLETE"
elif [ -f "$OUT/V5_P2_A1_R1_PAIR_LENGTH_AND_WINDOW_SHORTCUT_AUDIT_HOLD" ]; then
    cat "$OUT/V5_P2_A1_R1_PAIR_LENGTH_AND_WINDOW_SHORTCUT_AUDIT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
