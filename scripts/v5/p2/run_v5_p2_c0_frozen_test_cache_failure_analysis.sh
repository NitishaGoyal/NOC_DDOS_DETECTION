#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

B6="$REPO/reports/v5/p2_b6_one_shot_test_evaluation"
SCRIPT="$REPO/scripts/v5/p2/analyze_v5_p2_c0_frozen_test_cache.py"
OUT="$REPO/reports/v5/p2_c0_frozen_test_cache_failure_analysis"
LOG="$REPO/logs/v5/v5_p2_c0_frozen_test_cache_failure_analysis.log"

echo "===== V5 P2-C0 FROZEN TEST-CACHE FAILURE ANALYSIS ====="
echo "repo=$REPO"
echo "b6=$B6"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "analysis_source=B6_prediction_cache_and_pair_manifest_only"
echo "model_loaded=false"
echo "checkpoint_loaded=false"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo "test_inference_rerun=false"
echo

[ -d "$B6" ] || { echo "STOP: missing B6 directory: $B6"; exit 2; }
[ -f "$SCRIPT" ] || { echo "STOP: missing script: $SCRIPT"; exit 2; }
[ ! -e "$OUT" ] || { echo "STOP: output already exists: $OUT"; exit 2; }
[ ! -e "$LOG" ] || { echo "STOP: log already exists: $LOG"; exit 2; }

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
python -u "$SCRIPT" --b6-dir "$B6" --output-dir "$OUT" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

echo
echo "v5_p2_c0_status=$status"
if [ -f "$OUT/V5_P2_C0_FROZEN_TEST_CACHE_FAILURE_ANALYSIS_COMPLETE" ]; then
    cat "$OUT/V5_P2_C0_FROZEN_TEST_CACHE_FAILURE_ANALYSIS_COMPLETE"
elif [ -f "$OUT/V5_P2_C0_FROZEN_TEST_CACHE_FAILURE_ANALYSIS_HOLD" ]; then
    cat "$OUT/V5_P2_C0_FROZEN_TEST_CACHE_FAILURE_ANALYSIS_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi
exit "$status"
