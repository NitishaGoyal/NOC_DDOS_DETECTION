#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

B1="$REPO/reports/v5/p2_b1_training_protocol_lock"
SEED_REPORT_ROOT="$REPO/reports/v5/p2_b2_multi_seed_training"
SEED_MODEL_ROOT="$REPO/models/v5/p2_b2_multi_seed"
SCRIPT="$REPO/scripts/v5/p2/finalize_v5_p2_b2_multi_seed_training.py"

OUT="$REPO/reports/v5/p2_b2_multi_seed_training/finalization"
LOG="$REPO/logs/v5/v5_p2_b2_multi_seed_finalization.log"

echo "===== V5 P2-B2 MULTI-SEED FINALIZATION ====="
echo "repo=$REPO"
echo "b1=$B1"
echo "seed_report_root=$SEED_REPORT_ROOT"
echo "seed_model_root=$SEED_MODEL_ROOT"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "dataset_constructed=false"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo

[ -d "$B1" ] || {
    echo "STOP: missing B1 directory: $B1"
    exit 2
}
[ -f "$SCRIPT" ] || {
    echo "STOP: missing script: $SCRIPT"
    exit 2
}

for seed in 107 117 127; do
    [ -d "$SEED_REPORT_ROOT/seed_${seed}" ] || {
        echo "STOP: missing seed report directory: seed_${seed}"
        exit 2
    }
    [ -d "$SEED_MODEL_ROOT/seed_${seed}" ] || {
        echo "STOP: missing seed model directory: seed_${seed}"
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
    --b1-dir "$B1" \
    --seed-report-root "$SEED_REPORT_ROOT" \
    --seed-model-root "$SEED_MODEL_ROOT" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_b2_finalization_status=$status"

if [ -f "$OUT/V5_P2_B2_MULTI_SEED_TRAINING_COMPLETE" ]; then
    cat "$OUT/V5_P2_B2_MULTI_SEED_TRAINING_COMPLETE"
elif [ -f "$OUT/V5_P2_B2_MULTI_SEED_TRAINING_HOLD" ]; then
    cat "$OUT/V5_P2_B2_MULTI_SEED_TRAINING_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
