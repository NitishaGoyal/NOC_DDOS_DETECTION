#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p0_initial24_graph"
SCRIPT="$REPO/scripts/v5/common/inspect_v5_p0_a3_0_loader_interface.py"
OUT="$REPO/reports/v5/p0_a3_0_loader_interface"
LOG="$REPO/logs/v5/v5_p0_a3_0_loader_interface.log"

echo "===== V5 P0-A3-0 LOADER INTERFACE INSPECTION ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo

if [ ! -d "$DATA" ]; then
    echo "STOP: dataset root missing: $DATA"
    exit 2
fi

if [ ! -f "$SCRIPT" ]; then
    echo "STOP: inspection script missing: $SCRIPT"
    exit 2
fi

if [ -e "$OUT" ]; then
    echo "STOP: output directory already exists: $OUT"
    exit 2
fi

if [ -e "$LOG" ]; then
    echo "STOP: log already exists: $LOG"
    exit 2
fi

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --root "$DATA" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_a3_0_status=$status"

if [ -f "$OUT/V5_P0_A3_0_LOADER_INTERFACE_INSPECTION_COMPLETE" ]; then
    cat "$OUT/V5_P0_A3_0_LOADER_INTERFACE_INSPECTION_COMPLETE"
fi

exit "$status"
