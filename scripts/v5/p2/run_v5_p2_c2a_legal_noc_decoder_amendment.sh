#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

C2="$REPO/reports/v5/p2_c2_next_dataset_model_revision_contract"
SCRIPT="$REPO/scripts/v5/p2/freeze_v5_p2_c2a_legal_noc_decoder_amendment.py"
OUT="$REPO/reports/v5/p2_c2a_legal_noc_decoder_amendment"
LOG="$REPO/logs/v5/v5_p2_c2a_legal_noc_decoder_amendment.log"

echo "===== V5 P2-C2A LEGAL NOC DECODER AMENDMENT ====="
echo "repo=$REPO"
echo "c2=$C2"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "c2_modified=false"
echo "decoder_implementation_performed=false"
echo "p2_prediction_cache_accessed=false"
echo "p2_checkpoint_loaded=false"
echo "p2_test_directory_enumerated=false"
echo "p2_test_tensor_contents_accessed=false"
echo "p2_test_inference_rerun=false"
echo

[ -d "$C2" ] || { echo "STOP: missing C2 directory: $C2"; exit 2; }
[ -f "$SCRIPT" ] || { echo "STOP: missing amendment script: $SCRIPT"; exit 2; }
[ ! -e "$OUT" ] || { echo "STOP: output already exists: $OUT"; exit 2; }
[ ! -e "$LOG" ] || { echo "STOP: log already exists: $LOG"; exit 2; }

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --c2-dir "$C2" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_c2a_status=$status"

if [ -f "$OUT/V5_P2_C2A_LEGAL_NOC_DECODER_AMENDMENT_COMPLETE" ]; then
    cat "$OUT/V5_P2_C2A_LEGAL_NOC_DECODER_AMENDMENT_COMPLETE"
elif [ -f "$OUT/V5_P2_C2A_LEGAL_NOC_DECODER_AMENDMENT_HOLD" ]; then
    cat "$OUT/V5_P2_C2A_LEGAL_NOC_DECODER_AMENDMENT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
