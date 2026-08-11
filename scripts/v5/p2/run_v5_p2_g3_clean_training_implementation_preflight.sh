#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
G0="$REPO/reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol"
TOPOLOGY="$REPO/reports/v5/p2_g1a_r2a_canonical_static_topology_contract"
B0_R3="$REPO/reports/v5/p2_b0_r3_corrected_nontest_label_shortcut_audit"
G1_AGG="$REPO/reports/v5/p2_g1c_source_only_graph_matrix_aggregation"
G2_AGG="$REPO/reports/v5/p2_g2_direct_graph_matrix_aggregation"
PAIR_MANIFEST="$REPO/reports/v5/p2_a1_r2_pair_aligned_window_contract/V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
B3_MODEL="$REPO/src/models/v5_p2_b3_conv1d_only_count4.py"
TRAINER="$REPO/scripts/v5/p2/train_v5_p2_g3_role_aware_multilabel_single_run.py"
SCRIPT="$REPO/scripts/v5/p2/preflight_v5_p2_g3_clean_training_implementation.py"
OUT="$REPO/reports/v5/p2_g3_clean_training_implementation_preflight"
LOG="$REPO/logs/v5/v5_p2_g3_clean_training_implementation_preflight.log"
COMPLETE="$OUT/V5_P2_G3_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_COMPLETE"

if [ -f "$COMPLETE" ]; then
    echo "G3 clean preflight already complete"
    cat "$COMPLETE"
    exit 0
fi

for path in "$DATA" "$G0" "$TOPOLOGY" "$B0_R3" "$G1_AGG" "$G2_AGG"; do
    [ -d "$path" ] || { echo "STOP: missing directory: $path"; exit 2; }
done
for path in "$PAIR_MANIFEST" "$LOADER" "$B3_MODEL" "$TRAINER" "$SCRIPT"; do
    [ -f "$path" ] || { echo "STOP: missing file: $path"; exit 2; }
done
[ ! -e "$OUT" ] || { echo "STOP: partial preflight output exists: $OUT"; exit 2; }
[ ! -e "$LOG" ] || { echo "STOP: preflight log already exists: $LOG"; exit 2; }
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --root "$DATA" \
    --g0-dir "$G0" \
    --topology-dir "$TOPOLOGY" \
    --b0-r3-dir "$B0_R3" \
    --g1-aggregation-dir "$G1_AGG" \
    --g2-aggregation-dir "$G2_AGG" \
    --pair-manifest "$PAIR_MANIFEST" \
    --loader-path "$LOADER" \
    --b3-model-path "$B3_MODEL" \
    --trainer-path "$TRAINER" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "v5_p2_g3_clean_preflight_status=$status"
[ -f "$COMPLETE" ] && cat "$COMPLETE"
exit "$status"
