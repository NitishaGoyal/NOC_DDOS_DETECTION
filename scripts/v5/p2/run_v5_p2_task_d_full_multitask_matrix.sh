#!/usr/bin/env bash
set -uo pipefail
REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"
ONE_RUN="$REPO/scripts/v5/p2/run_v5_p2_task_d_full_multitask_one_run.sh"
AGGREGATE="$REPO/scripts/v5/p2/run_v5_p2_task_d_full_multitask_aggregation.sh"
PREFLIGHT_MARKER="$REPO/reports/v5/p2_task_d_full_multitask_preflight/V5_P2_TASK_D_FULL_MULTITASK_PREFLIGHT_COMPLETE"
LOCK_DIR="$REPO/reports/v5/.p2_task_d_full_multitask_matrix.lock"
[ -f "$ONE_RUN" ] || { echo "STOP: missing one-run launcher: $ONE_RUN"; exit 2; }
[ -f "$AGGREGATE" ] || { echo "STOP: missing aggregation launcher: $AGGREGATE"; exit 2; }
[ -f "$PREFLIGHT_MARKER" ] || { echo "STOP: Task-D preflight is not complete"; exit 2; }
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "STOP: Task-D matrix lock already exists: $LOCK_DIR"
    exit 2
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT INT TERM
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "===== V5 P2 TASK-D SERIAL 10-RUN FULL MULTITASK MATRIX ====="
echo "order=seed-major"
echo "seeds=107,117,127,137,147"
echo "candidates=conv1d,graphconv"
echo "graphconv_scope=TASK_D_REVIEWER_CHALLENGER_ONLY"
echo "loss_and_checkpoint_selection=frozen_B1_exact"
echo "threshold_tuning_during_training=false"
echo "concurrent_gpu_processes=1"
echo "architecture_selected=false"
echo "test_directory_enumerated=false"
echo "test_tensors_deserialized=false"
echo

for seed in 107 117 127 137 147; do
    for candidate in conv1d graphconv; do
        echo
        echo "===== MATRIX ITEM candidate=$candidate seed=$seed ====="
        bash "$ONE_RUN" "$candidate" "$seed"
        status=$?
        if [ "$status" -ne 0 ]; then
            echo "HOLD: matrix stopped at candidate=$candidate seed=$seed status=$status"
            exit "$status"
        fi
    done
done

echo
echo "===== ALL 10 TASK-D RUNS COMPLETE; AGGREGATING ====="
bash "$AGGREGATE"
status=$?
if [ "$status" -ne 0 ]; then
    echo "HOLD: all Task-D runs finished but aggregation failed"
    exit "$status"
fi
echo "V5_P2_TASK_D_FULL_MULTITASK_TRAINING_MATRIX_COMPLETE"
