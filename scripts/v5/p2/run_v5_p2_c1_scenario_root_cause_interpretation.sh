#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

C0="$REPO/reports/v5/p2_c0_frozen_test_cache_failure_analysis"
SCRIPT="$REPO/scripts/v5/p2/interpret_v5_p2_c1_scenario_root_causes.py"
OUT="$REPO/reports/v5/p2_c1_scenario_root_cause_interpretation"
LOG="$REPO/logs/v5/v5_p2_c1_scenario_root_cause_interpretation.log"

echo "===== V5 P2-C1 SCENARIO + ROOT-CAUSE INTERPRETATION ====="
echo "repo=$REPO"
echo "c0=$C0"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "analysis_source=P2_C0_outputs_only"
echo "prediction_cache_opened=false"
echo "model_loaded=false"
echo "checkpoint_loaded=false"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo "test_inference_rerun=false"
echo

[ -d "$C0" ] || {
    echo "STOP: missing C0 directory: $C0"
    exit 2
}
[ -f "$SCRIPT" ] || {
    echo "STOP: missing interpretation script: $SCRIPT"
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
    --c0-dir "$C0" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_c1_status=$status"

if [ -f "$OUT/V5_P2_C1_SCENARIO_AND_ROOT_CAUSE_INTERPRETATION_COMPLETE" ]; then
    cat "$OUT/V5_P2_C1_SCENARIO_AND_ROOT_CAUSE_INTERPRETATION_COMPLETE"
elif [ -f "$OUT/V5_P2_C1_SCENARIO_AND_ROOT_CAUSE_INTERPRETATION_HOLD" ]; then
    cat "$OUT/V5_P2_C1_SCENARIO_AND_ROOT_CAUSE_INTERPRETATION_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
