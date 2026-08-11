#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p0_initial24_graph"
B8="$REPO/reports/v5/p0_b8_freeze_b3_conv1d_only_architecture"
WRAPPER="$REPO/scripts/v5/common/v5_p0_contract_loader.py"
B3_SCRIPT="$REPO/scripts/v5/baselines/train_v5_p0_b3_conv1d_only.py"
SCRIPT="$REPO/scripts/v5/final/freeze_v5_p0_c0_final_b3_protocol.py"

OUT="$REPO/reports/v5/p0_c0_final_b3_training_protocol_lock"
LOG="$REPO/logs/v5/v5_p0_c0_final_b3_training_protocol_lock.log"

echo "===== V5 P0-C0 FINAL B3 TRAINING PROTOCOL LOCK ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "b8=$B8"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "training_performed=false"
echo "test_split_constructed=false"
echo "test_split_accessed=false"
echo

for path in "$DATA" "$B8"; do
    [ -d "$path" ] || {
        echo "STOP: missing directory: $path"
        exit 2
    }
done

for path in "$WRAPPER" "$B3_SCRIPT" "$SCRIPT"; do
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
    --b8-dir "$B8" \
    --data-root "$DATA" \
    --wrapper "$WRAPPER" \
    --b3-script "$B3_SCRIPT" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_c0_status=$status"

if [ -f "$OUT/V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK_COMPLETE" ]; then
    cat "$OUT/V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK_COMPLETE"
elif [ -f "$OUT/V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK_HOLD" ]; then
    cat "$OUT/V5_P0_C0_FINAL_B3_TRAINING_PROTOCOL_LOCK_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
