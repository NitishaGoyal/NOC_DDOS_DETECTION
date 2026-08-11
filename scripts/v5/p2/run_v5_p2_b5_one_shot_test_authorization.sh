#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

B1="$REPO/reports/v5/p2_b1_training_protocol_lock"
B3="$REPO/reports/v5/p2_b3_seed_checkpoint_selection"
B4="$REPO/reports/v5/p2_b4_validation_threshold_tuning"

SCRIPT="$REPO/scripts/v5/p2/authorize_v5_p2_b5_one_shot_test.py"
OUT="$REPO/reports/v5/p2_b5_one_shot_test_authorization"
LOG="$REPO/logs/v5/v5_p2_b5_one_shot_test_authorization.log"

echo "===== V5 P2-B5 ONE-SHOT TEST AUTHORIZATION ====="
echo "repo=$REPO"
echo "b1=$B1"
echo "b3=$B3"
echo "b4=$B4"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "selected_checkpoint_deserialized=false"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo "test_evaluation_performed=false"
echo "requested_authorization_count=1"
echo

for path in "$B1" "$B3" "$B4"; do
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
    --b1-dir "$B1" \
    --b3-dir "$B3" \
    --b4-dir "$B4" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_b5_status=$status"

if [ -f "$OUT/V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION_COMPLETE" ]; then
    cat "$OUT/V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION_COMPLETE"
elif [ -f "$OUT/V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION_HOLD" ]; then
    cat "$OUT/V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
