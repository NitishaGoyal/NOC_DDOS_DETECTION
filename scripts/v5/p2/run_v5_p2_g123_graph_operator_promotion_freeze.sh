#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO"
source "$REPO/.venv/bin/activate"
SCRIPT="$REPO/scripts/v5/p2/freeze_v5_p2_g123_graph_operator_promotion.py"
OUT="$REPO/reports/v5/p2_g123_graph_operator_promotion"
[ -f "$SCRIPT" ] || { echo "STOP: missing script: $SCRIPT"; exit 2; }
[ ! -e "$OUT" ] || { echo "STOP: output already exists: $OUT"; exit 2; }
python -u "$SCRIPT" --repo "$REPO" --output-dir "$OUT"
