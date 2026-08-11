#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

B8="$REPO/reports/v5/p0_b8_freeze_b3_conv1d_only_architecture"
C0="$REPO/reports/v5/p0_c0_final_b3_training_protocol_lock"
C1_REPORT="$REPO/reports/v5/p0_c1_b3_multi_seed_validation_training"
C1_MODEL="$REPO/models/v5/p0_c1_b3_multi_seed_validation_training"
C2_REPORT="$REPO/reports/v5/p0_c2_final_checkpoint_threshold_freeze"
C2_MODEL="$REPO/models/v5/p0_c2_final_checkpoint_threshold_freeze"
C3_REPORT="$REPO/reports/v5/p0_c3_one_shot_locked_test_evaluation"
C3_MODEL="$REPO/models/v5/p0_c3_one_shot_locked_test_evaluation"

SCRIPT="$REPO/scripts/v5/final/archive_v5_p0_and_handoff_to_p2.py"
OUT="$REPO/reports/v5/p0_final_results_archive_and_p2_handoff"
LOG="$REPO/logs/v5/v5_p0_final_results_archive_and_p2_handoff.log"

echo "===== V5 P0 FINAL ARCHIVE + P2 HANDOFF ====="
echo "repo=$REPO"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "P1_executed=false"
echo "P2_next=true"
echo "training_performed=false"
echo "dataset_constructed=false"
echo "test_evaluation_performed=false"
echo

for path in \
    "$B8" "$C0" "$C1_REPORT" "$C1_MODEL" \
    "$C2_REPORT" "$C2_MODEL" "$C3_REPORT" "$C3_MODEL"
do
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
    --b8-dir "$B8" \
    --c0-dir "$C0" \
    --c1-report-dir "$C1_REPORT" \
    --c1-model-dir "$C1_MODEL" \
    --c2-report-dir "$C2_REPORT" \
    --c2-model-dir "$C2_MODEL" \
    --c3-report-dir "$C3_REPORT" \
    --c3-model-dir "$C3_MODEL" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_final_archive_p2_handoff_status=$status"

if [ -f "$OUT/V5_P0_FINAL_RESULTS_ARCHIVE_AND_P2_HANDOFF_COMPLETE" ]; then
    cat "$OUT/V5_P0_FINAL_RESULTS_ARCHIVE_AND_P2_HANDOFF_COMPLETE"
elif [ -f "$OUT/V5_P0_FINAL_RESULTS_ARCHIVE_AND_P2_HANDOFF_HOLD" ]; then
    cat "$OUT/V5_P0_FINAL_RESULTS_ARCHIVE_AND_P2_HANDOFF_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
