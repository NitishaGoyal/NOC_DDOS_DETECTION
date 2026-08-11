#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

G0="$REPO/reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol"
R2_HOLD="$REPO/reports/v5/p2_g1a_r2_static_edge_index_resolution"
LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
SCRIPT="$REPO/scripts/v5/p2/freeze_v5_p2_g1a_r2a_canonical_topology_contract.py"
OUT="$REPO/reports/v5/p2_g1a_r2a_canonical_static_topology_contract"
LOG="$REPO/logs/v5/v5_p2_g1a_r2a_canonical_static_topology_contract.log"

echo "===== V5 P2-G1A-R2A CANONICAL TOPOLOGY CONTRACT ====="
echo "repo=$REPO"
echo "g0=$G0"
echo "r2_hold=$R2_HOLD"
echo "loader=$LOADER"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "historical_g1a_hold_preserved=true"
echo "historical_g1a_r1_hold_preserved=true"
echo "historical_g1a_r2_hold_preserved=true"
echo "p2_test_directory_enumerated=false"
echo "p2_test_tensors_deserialized=false"
echo "checkpoint_loaded=false"
echo "b4_validation_cache_accessed=false"
echo "b6_test_cache_accessed=false"
echo "training_performed=false"
echo "optimization_steps=0"
echo "architecture_selected=false"
echo "quantization_performed=false"
echo "rtl_generated=false"
echo "legal_decoder_implemented=false"
echo

for path in "$G0" "$R2_HOLD"; do
    [ -d "$path" ] || {
        echo "STOP: missing directory: $path"
        exit 2
    }
done

for path in "$LOADER" "$SCRIPT"; do
    [ -f "$path" ] || {
        echo "STOP: missing file: $path"
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
    --repo "$REPO" \
    --g0-dir "$G0" \
    --r2-hold-dir "$R2_HOLD" \
    --loader "$LOADER" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_g1a_r2a_status=$status"

if [ -f "$OUT/V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT_COMPLETE" ]; then
    cat "$OUT/V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT_COMPLETE"
elif [ -f "$OUT/V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT_HOLD" ]; then
    cat "$OUT/V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
