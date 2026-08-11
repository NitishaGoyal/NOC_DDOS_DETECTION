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
WRAPPER="$REPO/scripts/v5/common/v5_p0_contract_loader.py"
SCRIPT="$REPO/scripts/v5/baselines/train_v5_p0_b5_conv1d_gcn.py"
OUT="$REPO/reports/v5/p0_b5_conv1d_gcn"
MODEL_DIR="$REPO/models/v5/p0_b5_conv1d_gcn"
LOG="$REPO/logs/v5/v5_p0_b5_conv1d_gcn.log"

echo "===== V5 P0-B5 CONV1D + GCN ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "a2=$A2"
echo "a2_1=$A2_1"
echo "a3=$A3"
echo "r1c=$R1C"
echo "b1=$B1"
echo "b2=$B2"
echo "b3=$B3"
echo "b4=$B4"
echo "wrapper=$WRAPPER"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "model_dir=$MODEL_DIR"
echo "log=$LOG"
echo

for path in \
    "$DATA" "$A2" "$A2_1" "$A3" \
    "$R1C" "$B1" "$B2" "$B3" "$B4"
do
    if [ ! -d "$path" ]; then
        echo "STOP: required directory missing: $path"
        exit 2
    fi
done

for path in "$WRAPPER" "$SCRIPT"; do
    if [ ! -f "$path" ]; then
        echo "STOP: required file missing: $path"
        exit 2
    fi
done

if [ -e "$OUT" ]; then
    echo "STOP: output directory already exists: $OUT"
    exit 2
fi
if [ -e "$MODEL_DIR" ]; then
    echo "STOP: model directory already exists: $MODEL_DIR"
    exit 2
fi
if [ -e "$LOG" ]; then
    echo "STOP: log already exists: $LOG"
    exit 2
fi

mkdir -p \
    "$(dirname "$OUT")" \
    "$(dirname "$MODEL_DIR")" \
    "$(dirname "$LOG")"

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
    --wrapper "$WRAPPER" \
    --output-dir "$OUT" \
    --model-dir "$MODEL_DIR" \
    --epochs 100 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --seed 101 \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p0_b5_status=$status"

if [ -f "$OUT/V5_P0_B5_CONV1D_GCN_COMPLETE" ]; then
    cat "$OUT/V5_P0_B5_CONV1D_GCN_COMPLETE"
elif [ -f "$OUT/V5_P0_B5_CONV1D_GCN_HOLD" ]; then
    cat "$OUT/V5_P0_B5_CONV1D_GCN_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
