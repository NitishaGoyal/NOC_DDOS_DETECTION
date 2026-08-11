#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
B1="$REPO/reports/v5/p2_b1_training_protocol_lock"
B3="$REPO/reports/v5/p2_b3_seed_checkpoint_selection"
B4="$REPO/reports/v5/p2_b4_validation_threshold_tuning"
B5="$REPO/reports/v5/p2_b5_one_shot_test_authorization"

LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
MODEL_SOURCE="$REPO/src/models/v5_p2_b3_conv1d_only_count4.py"
SCRIPT="$REPO/scripts/v5/p2/evaluate_v5_p2_b6_one_shot_test.py"

OUT="$REPO/reports/v5/p2_b6_one_shot_test_evaluation"
LOG="$REPO/logs/v5/v5_p2_b6_one_shot_test_evaluation.log"

echo "===== V5 P2-B6 ONE-SHOT BLIND TEST EVALUATION ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "b1=$B1"
echo "b3=$B3"
echo "b4=$B4"
echo "b5=$B5"
echo "loader=$LOADER"
echo "model_source=$MODEL_SOURCE"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "authorization_id=95da82aebaf1b61b13ce4bf31346b08500037841369cfd3fe5dba0965dff24e2"
echo "authorized_evaluation_count=1"
echo "successful_rerun_authorized=false"
echo "NOTE: this command irreversibly consumes the one-shot authorization"
echo

for path in "$DATA" "$B1" "$B3" "$B4" "$B5"; do
    [ -d "$path" ] || {
        echo "STOP: missing directory: $path"
        exit 2
    }
done

for path in "$LOADER" "$MODEL_SOURCE" "$SCRIPT"; do
    [ -f "$path" ] || {
        echo "STOP: missing file: $path"
        exit 2
    }
done

[ ! -e "$OUT" ] || {
    echo "STOP: one-shot output already exists: $OUT"
    exit 2
}
[ ! -e "$LOG" ] || {
    echo "STOP: one-shot log already exists: $LOG"
    exit 2
}
[ ! -e "$B5/V5_P2_B5_ONE_SHOT_TEST_AUTHORIZATION_CONSUMED.json" ] || {
    echo "STOP: B5 authorization is already consumed"
    exit 2
}

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --root "$DATA" \
    --b1-dir "$B1" \
    --b3-dir "$B3" \
    --b4-dir "$B4" \
    --b5-dir "$B5" \
    --loader-path "$LOADER" \
    --model-source-path "$MODEL_SOURCE" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_b6_status=$status"

if [ -f "$OUT/V5_P2_B6_ONE_SHOT_TEST_EVALUATION_COMPLETE" ]; then
    cat "$OUT/V5_P2_B6_ONE_SHOT_TEST_EVALUATION_COMPLETE"
elif [ -f "$OUT/V5_P2_B6_ONE_SHOT_TEST_EVALUATION_IRREVERSIBLE_FAILURE" ]; then
    cat "$OUT/V5_P2_B6_ONE_SHOT_TEST_EVALUATION_IRREVERSIBLE_FAILURE"
elif [ -f "$OUT/V5_P2_B6_ONE_SHOT_TEST_EVALUATION_HOLD" ]; then
    cat "$OUT/V5_P2_B6_ONE_SHOT_TEST_EVALUATION_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
