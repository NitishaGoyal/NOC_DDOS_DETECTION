#!/usr/bin/env bash
set -uo pipefail

REPO="$HOME/research/projects/GNN-2d"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/graph_dataset/paper1_temporal_graphs_ports_v4_all16_chrono_memmap"

TRAINER="$REPO/scripts/v4/train/train_v4_a3_sourcepreserve.py"
MODEL_SOURCE="$REPO/scripts/v4/common/v4_a3_sourcepreserve.py"
BASE_CHECKPOINT="$REPO/models/v4/v4_a3_sourcepreserve_countaware_rank_mask_seed7/best_model.pt"

SLOT_MODULE="$REPO/scripts/v4/a4a/v4_a4a_slot_model.py"
SMOKE_SCRIPT="$REPO/scripts/v4/a4a/train_v4_a4a_1b_slot_smoke.py"
CONTRACT_FREEZE_SCRIPT="$REPO/scripts/v4/a4a/v4_a4a_1c_freeze_full_training_contract.py"
RESOURCE_PROBE_SCRIPT="$REPO/scripts/v4/a4a/v4_a4a_1d_batch512_resource_probe.py"
PRIMARY_TRAINER="$REPO/scripts/v4/a4a/train_v4_a4a_2a_slot_full.py"

CONTRACT_DIR="$REPO/reports/v4/stage1_a4a/a4a_1c_full_training_contract"
PROBE_DIR="$REPO/reports/v4/stage1_a4a/a4a_1d_batch512_resource_probe"
SOURCE_AUDIT="$REPO/reports/v4/stage1_a4a/a4a_2a_primary_trainer_source_audit"

TRAIN_OUT="$REPO/models/v4/v4_a4a_slot_full_seed7"
LOG="$REPO/logs/v4/v4_a4a_slot_full_seed7.log"

echo "===== V4 A4a-2A PRIMARY TRAINING ====="
echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "training_output=$TRAIN_OUT"
echo "log=$LOG"
echo

python -u "$PRIMARY_TRAINER" \
  --mode train \
  --repo "$REPO" \
  --data-dir "$DATA" \
  --trainer "$TRAINER" \
  --model-source "$MODEL_SOURCE" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --slot-module "$SLOT_MODULE" \
  --smoke-script "$SMOKE_SCRIPT" \
  --contract-freeze-script "$CONTRACT_FREEZE_SCRIPT" \
  --resource-probe-script "$RESOURCE_PROBE_SCRIPT" \
  --contract-dir "$CONTRACT_DIR" \
  --probe-dir "$PROBE_DIR" \
  --source-audit-dir "$SOURCE_AUDIT" \
  --out-dir "$TRAIN_OUT" \
  2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "a4a_2a_train_status=$status"
echo "finished_at=$(date --iso-8601=seconds)"

if [ -f "$TRAIN_OUT/V4_A4A_2A_PRIMARY_TRAINING_PASS" ]; then
  echo "===== FINAL MARKER ====="
  cat "$TRAIN_OUT/V4_A4A_2A_PRIMARY_TRAINING_PASS"
elif [ -f "$TRAIN_OUT/last_model.pt" ]; then
  echo "Training stopped before completion."
  echo "A resumable last_model.pt exists."
else
  echo "Training stopped without a resumable checkpoint."
fi

exit "$status"
