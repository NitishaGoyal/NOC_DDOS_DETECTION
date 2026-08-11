#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO"
source "$REPO/.venv/bin/activate"
SCRIPT="$REPO/scripts/v5/p2/aggregate_v5_p2_task_d_full_multitask_matrix.py"
TRAINING="$REPO/reports/v5/p2_task_d_full_multitask_training"
PROMOTION="$REPO/reports/v5/p2_g123_graph_operator_promotion"
OUT="$REPO/reports/v5/p2_task_d_full_multitask_matrix_aggregation"
[ -f "$SCRIPT" ] || { echo "STOP: missing script: $SCRIPT"; exit 2; }
[ -d "$TRAINING" ] || { echo "STOP: missing training reports: $TRAINING"; exit 2; }
[ ! -e "$OUT" ] || { echo "STOP: output already exists: $OUT"; exit 2; }
python -u "$SCRIPT" --training-root "$TRAINING" --promotion-dir "$PROMOTION" --output-dir "$OUT"
