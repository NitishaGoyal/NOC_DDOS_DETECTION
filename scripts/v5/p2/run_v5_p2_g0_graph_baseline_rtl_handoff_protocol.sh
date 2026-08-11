#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

C2A="$REPO/reports/v5/p2_c2a_legal_noc_decoder_amendment"
SCRIPT="$REPO/scripts/v5/p2/freeze_v5_p2_g0_graph_baseline_rtl_handoff_protocol.py"
OUT="$REPO/reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol"
LOG="$REPO/logs/v5/v5_p2_g0_graph_baseline_rtl_handoff_protocol.log"

echo "===== V5 P2-G0 GRAPH BASELINE + RTL HANDOFF PROTOCOL ====="
echo "repo=$REPO"
echo "c2a=$C2A"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "training_performed=false"
echo "inference_performed=false"
echo "b4_validation_cache_accessed=false"
echo "b6_test_cache_accessed=false"
echo "p2_test_directory_enumerated=false"
echo "p2_test_tensor_contents_accessed=false"
echo "p2_test_inference_rerun=false"
echo "rtl_generated=false"
echo "legal_decoder_implemented=false"
echo

[ -d "$C2A" ] || {
    echo "STOP: missing C2A directory: $C2A"
    exit 2
}
[ -f "$SCRIPT" ] || {
    echo "STOP: missing G0 script: $SCRIPT"
    exit 2
}
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
    --c2a-dir "$C2A" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_g0_status=$status"

if [ -f "$OUT/V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL_COMPLETE" ]; then
    cat "$OUT/V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL_COMPLETE"
elif [ -f "$OUT/V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL_HOLD" ]; then
    cat "$OUT/V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
