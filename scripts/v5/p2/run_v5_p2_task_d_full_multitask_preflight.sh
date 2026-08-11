#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO"
source "$REPO/.venv/bin/activate"
PROMOTION="$REPO/reports/v5/p2_g123_graph_operator_promotion/V5_P2_G123_GRAPH_OPERATOR_PROMOTION_COMPLETE"
SCRIPT="$REPO/scripts/v5/p2/preflight_v5_p2_task_d_full_multitask.py"
OUT="$REPO/reports/v5/p2_task_d_full_multitask_preflight"
[ -f "$PROMOTION" ] || { echo "STOP: graph-operator promotion review is not frozen"; exit 2; }
[ -f "$SCRIPT" ] || { echo "STOP: missing script: $SCRIPT"; exit 2; }
[ ! -e "$OUT" ] || { echo "STOP: output already exists: $OUT"; exit 2; }
python -u "$SCRIPT" --repo "$REPO" --output-dir "$OUT"
