#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

A2="$REPO/reports/v5/p0_a2_feature_contract"
A2_1="$REPO/reports/v5/p0_a2_1_mask_recoverability_audit"
A3="$REPO/reports/v5/p0_a3_loader_contract_smoke"
R1C="$REPO/reports/v5/p0_b0_r1c_final_quarantine_release"
B1="$REPO/reports/v5/p0_b1_static_final_epoch_mlp"
B2="$REPO/reports/v5/p0_b2_temporal_mean_pooling"
B3="$REPO/reports/v5/p0_b3_conv1d_only"
B4="$REPO/reports/v5/p0_b4_gcn_only"
B5="$REPO/reports/v5/p0_b5_conv1d_gcn"
B6="$REPO/reports/v5/p0_b6_dual_temporal_fusion_no_gcn"
B7A="$REPO/reports/v5/p0_b7a_directional_gated_graph_diagnostic"
B7B="$REPO/reports/v5/p0_b7b_directional_gated_graph_multi_seed"

WRAPPER="$REPO/scripts/v5/common/v5_p0_contract_loader.py"
B3_SCRIPT="$REPO/scripts/v5/baselines/train_v5_p0_b3_conv1d_only.py"
SCRIPT="$REPO/scripts/v5/freeze/freeze_v5_p0_b8_b3_architecture.py"
OUT="$REPO/reports/v5/p0_b8_freeze_b3_conv1d_only_architecture"
LOG="$REPO/logs/v5/v5_p0_b8_freeze_b3_conv1d_only_architecture.log"

echo "===== V5 P0-B8 FREEZE B3 ARCHITECTURE ====="
echo "repo=$REPO"
echo "output=$OUT"
echo "log=$LOG"
echo "training_performed=false"
echo "test_split_accessed=false"
echo

for path in \
  "$A2" "$A2_1" "$A3" "$R1C" "$B1" "$B2" \
  "$B3" "$B4" "$B5" "$B6" "$B7A" "$B7B"
do
    [ -d "$path" ] || { echo "STOP: missing directory: $path"; exit 2; }
done

for path in "$WRAPPER" "$B3_SCRIPT" "$SCRIPT"; do
    [ -f "$path" ] || { echo "STOP: missing file: $path"; exit 2; }
done

[ ! -e "$OUT" ] || { echo "STOP: output exists: $OUT"; exit 2; }
[ ! -e "$LOG" ] || { echo "STOP: log exists: $LOG"; exit 2; }

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
  --a2-dir "$A2" \
  --a2-1-dir "$A2_1" \
  --a3-dir "$A3" \
  --r1c-dir "$R1C" \
  --b1-dir "$B1" \
  --b2-dir "$B2" \
  --b3-dir "$B3" \
  --b4-dir "$B4" \
  --b5-dir "$B5" \
  --b6-dir "$B6" \
  --b7a-dir "$B7A" \
  --b7b-dir "$B7B" \
  --wrapper "$WRAPPER" \
  --b3-script "$B3_SCRIPT" \
  --output-dir "$OUT" \
  2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
echo
echo "v5_p0_b8_status=$status"

if [ -f "$OUT/V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_COMPLETE" ]; then
    cat "$OUT/V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_COMPLETE"
elif [ -f "$OUT/V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_HOLD" ]; then
    cat "$OUT/V5_P0_B8_FREEZE_B3_CONV1D_ONLY_ARCHITECTURE_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
