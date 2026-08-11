#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

REPORTS_ROOT="$REPO/reports/v5/p2_g2_direct_graph_training"
G1_AGG="$REPO/reports/v5/p2_g1c_source_only_graph_matrix_aggregation"
G1_FREEZE="$REPO/reports/v5/p2_g1c_frozen_validation_result"
OUT="$REPO/reports/v5/p2_g2_direct_graph_matrix_aggregation"
LOG="$REPO/logs/v5/v5_p2_g2_direct_graph_matrix_aggregation.log"
SCRIPT="$REPO/scripts/v5/p2/aggregate_v5_p2_g2_direct_graph_training_matrix.py"
COMPLETE="$OUT/V5_P2_G2_DIRECT_GRAPH_MATRIX_AGGREGATION_COMPLETE"

if [ -f "$COMPLETE" ]; then
    echo "G2 aggregation already complete"
    cat "$COMPLETE"
    exit 0
fi
[ -d "$REPORTS_ROOT" ] || { echo "STOP: missing reports root: $REPORTS_ROOT"; exit 2; }
[ -d "$G1_AGG" ] || { echo "STOP: missing frozen G1 aggregation: $G1_AGG"; exit 2; }
[ -d "$G1_FREEZE" ] || { echo "STOP: missing frozen G1 decision: $G1_FREEZE"; exit 2; }
[ -f "$SCRIPT" ] || { echo "STOP: missing script: $SCRIPT"; exit 2; }
[ ! -e "$OUT" ] || { echo "STOP: partial aggregation output exists: $OUT"; exit 2; }
[ ! -e "$LOG" ] || { echo "STOP: aggregation log already exists: $LOG"; exit 2; }
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --reports-root "$REPORTS_ROOT" \
    --g1-aggregation-dir "$G1_AGG" \
    --g1-freeze-dir "$G1_FREEZE" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "v5_p2_g2_aggregation_status=$status"
[ -f "$COMPLETE" ] && cat "$COMPLETE"
exit "$status"
