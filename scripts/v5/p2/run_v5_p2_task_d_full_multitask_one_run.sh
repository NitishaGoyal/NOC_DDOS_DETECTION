#!/usr/bin/env bash
set -uo pipefail

CANDIDATE="${1:-}"
SEED="${2:-}"
case "$CANDIDATE" in conv1d|graphconv) ;; *) echo "Usage: $0 {conv1d|graphconv} {107|117|127|137|147}"; exit 2 ;; esac
case "$SEED" in 107|117|127|137|147) ;; *) echo "Usage: $0 {conv1d|graphconv} {107|117|127|137|147}"; exit 2 ;; esac

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
B1="$REPO/reports/v5/p2_b1_training_protocol_lock"
B0="$REPO/reports/v5/p2_b0_r3_corrected_nontest_label_shortcut_audit"
PROMOTION="$REPO/reports/v5/p2_g123_graph_operator_promotion"
PREFLIGHT="$REPO/reports/v5/p2_task_d_full_multitask_preflight"
TOPOLOGY="$REPO/reports/v5/p2_g1a_r2a_canonical_static_topology_contract"
PAIR_MANIFEST="$REPO/reports/v5/p2_a1_r2_pair_aligned_window_contract/V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
B3_MODEL="$REPO/src/models/v5_p2_b3_conv1d_only_count4.py"
TASK_D_MODEL="$REPO/src/models/v5_p2_task_d_full_multitask_count4.py"
B2_TRAIN="$REPO/scripts/v5/p2/train_v5_p2_b2_single_seed.py"
SCRIPT="$REPO/scripts/v5/p2/train_v5_p2_task_d_full_multitask_single_run.py"

MODEL_DIR="$REPO/models/v5/p2_task_d_full_multitask/$CANDIDATE/seed_${SEED}"
REPORT_DIR="$REPO/reports/v5/p2_task_d_full_multitask_training/$CANDIDATE/seed_${SEED}"
LOG="$REPO/logs/v5/v5_p2_task_d_${CANDIDATE}_seed_${SEED}.log"
COMPLETE_MARKER="$REPORT_DIR/V5_P2_TASK_D_FULL_MULTITASK_SINGLE_RUN_COMPLETE"
HOLD_MARKER="$REPORT_DIR/V5_P2_TASK_D_FULL_MULTITASK_SINGLE_RUN_HOLD"

if [ -f "$COMPLETE_MARKER" ]; then
    echo "ALREADY COMPLETE: $CANDIDATE seed $SEED"
    cat "$COMPLETE_MARKER"
    exit 0
fi
for path in "$DATA" "$B1" "$B0" "$PROMOTION" "$PREFLIGHT" "$TOPOLOGY"; do
    [ -d "$path" ] || { echo "STOP: missing directory: $path"; exit 2; }
done
for path in "$PAIR_MANIFEST" "$LOADER" "$B3_MODEL" "$TASK_D_MODEL" "$B2_TRAIN" "$SCRIPT"; do
    [ -f "$path" ] || { echo "STOP: missing file: $path"; exit 2; }
done
if [ -e "$MODEL_DIR" ] || [ -e "$REPORT_DIR" ] || [ -e "$LOG" ]; then
    echo "STOP: partial or pre-existing output exists for $CANDIDATE seed $SEED"
    echo "model_dir=$MODEL_DIR"
    echo "report_dir=$REPORT_DIR"
    echo "log=$LOG"
    [ -f "$HOLD_MARKER" ] && cat "$HOLD_MARKER"
    exit 2
fi
mkdir -p "$(dirname "$MODEL_DIR")" "$(dirname "$REPORT_DIR")" "$(dirname "$LOG")"

echo "===== V5 P2 TASK-D FULL MULTITASK RUN ====="
echo "candidate=$CANDIDATE"
echo "seed=$SEED"
echo "execution=serial_one_model_at_a_time"
echo "b1_protocol=reused_exactly"
echo "automatic_mixed_precision=false"
echo "threshold_tuning_performed=false"
echo "architecture_selected=false"
echo "test_directory_enumerated=false"
echo "test_tensors_deserialized=false"
echo

python -u "$SCRIPT" \
    --root "$DATA" \
    --b1-dir "$B1" \
    --b0-r3-dir "$B0" \
    --promotion-dir "$PROMOTION" \
    --preflight-dir "$PREFLIGHT" \
    --topology-dir "$TOPOLOGY" \
    --pair-manifest "$PAIR_MANIFEST" \
    --loader-path "$LOADER" \
    --b3-model-path "$B3_MODEL" \
    --task-d-model-path "$TASK_D_MODEL" \
    --b2-training-script-path "$B2_TRAIN" \
    --candidate "$CANDIDATE" \
    --seed "$SEED" \
    --model-dir "$MODEL_DIR" \
    --report-dir "$REPORT_DIR" \
    2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo
echo "v5_p2_task_d_${CANDIDATE}_seed_${SEED}_status=$status"
if [ -f "$COMPLETE_MARKER" ]; then
    cat "$COMPLETE_MARKER"
elif [ -f "$HOLD_MARKER" ]; then
    cat "$HOLD_MARKER"
else
    echo "NO FINAL MARKER FOUND"
fi
exit "$status"
