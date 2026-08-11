#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p0_initial24_graph"
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
WRAPPER="$REPO/scripts/v5/common/v5_p0_contract_loader.py"
FROZEN_MODULE="$REPO/scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic_b7b.py"
SCRIPT="$REPO/scripts/v5/baselines/run_v5_p0_b7b_multi_seed.py"
OUT="$REPO/reports/v5/p0_b7b_directional_gated_graph_multi_seed"
MODEL_DIR="$REPO/models/v5/p0_b7b_directional_gated_graph_multi_seed"
LOG="$REPO/logs/v5/v5_p0_b7b_directional_gated_graph_multi_seed.log"

echo "===== V5 P0-B7B PAIRED MULTI-SEED CONFIRMATION ====="
echo "repo=$REPO"
echo "seeds=7,17,27"
echo "output=$OUT"
echo "model_dir=$MODEL_DIR"
echo "log=$LOG"
echo

for path in "$DATA" "$A2" "$A2_1" "$A3" "$R1C" "$B1" "$B2" "$B3" "$B4" "$B5" "$B6" "$B7A"; do
    if [ ! -d "$path" ]; then
        echo "STOP: required directory missing: $path"
        exit 2
    fi
done
for path in "$WRAPPER" "$FROZEN_MODULE" "$SCRIPT"; do
    if [ ! -f "$path" ]; then
        echo "STOP: required file missing: $path"
        exit 2
    fi
done
if [ -e "$OUT" ] || [ -e "$MODEL_DIR" ] || [ -e "$LOG" ]; then
    echo "STOP: B7B output, model, or log path already exists"
    ls -ld "$OUT" "$MODEL_DIR" "$LOG" 2>/dev/null
    exit 2
fi

mkdir -p "$(dirname "$OUT")" "$(dirname "$MODEL_DIR")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --root "$DATA" \
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
    --wrapper "$WRAPPER" \
    --frozen-module "$FROZEN_MODULE" \
    --output-dir "$OUT" \
    --model-dir "$MODEL_DIR" \
    --epochs 100 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
echo
echo "v5_p0_b7b_status=$status"
if [ -f "$OUT/V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION_COMPLETE" ]; then
    cat "$OUT/V5_P0_B7B_DIRECTIONAL_GATED_GRAPH_MULTI_SEED_CONFIRMATION_COMPLETE"
elif [ -f "$OUT/V5_P0_B7B_HOLD" ]; then
    cat "$OUT/V5_P0_B7B_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi
exit "$status"
