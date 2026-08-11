#!/usr/bin/env bash
set -uo pipefail

REPO="$HOME/research/projects/GNN-2d"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/graph_dataset/paper1_temporal_graphs_ports_v4_all16_chrono_memmap"
TRAINING_DIR="$REPO/models/v4/v4_a4a_slot_full_seed7"

PRIMARY_TRAINER="$REPO/scripts/v4/a4a/train_v4_a4a_2a_slot_full.py"
ORIGINAL_TRAINER="$REPO/scripts/v4/train/train_v4_a3_sourcepreserve.py"
MODEL_SOURCE="$REPO/scripts/v4/common/v4_a3_sourcepreserve.py"
BASE_CHECKPOINT="$REPO/models/v4/v4_a3_sourcepreserve_countaware_rank_mask_seed7/best_model.pt"
SLOT_MODULE="$REPO/scripts/v4/a4a/v4_a4a_slot_model.py"

PREFLIGHT_SCRIPT="$REPO/scripts/v4/a4a/v4_a4a_2b0_validation_selection_preflight.py"
PREFLIGHT_DIR="$REPO/reports/v4/stage1_a4a/a4a_2b0_validation_selection_preflight"

FULL_TRAINING_CONTRACT="$REPO/reports/v4/stage1_a4a/a4a_1c_full_training_contract/A4A_1C_FULL_TRAINING_CONTRACT.json"
SLOT_CONTRACT="$REPO/reports/v4/stage1_a4a/a4a_0s_slot_contract_and_implementation_audit/A4A_SLOT_D0_FROZEN_CONTRACT.json"

FROZEN_L2_POLICY="$REPO/reports/v4/stage1_a3/a3_11_temporal_aggregation_precision_first/03_l2_graph_gated_refinement/FROZEN_L2_POLICY.json"
FROZEN_OPERATIONAL_POLICY="$REPO/reports/v4/stage1_a3/a3_11_temporal_aggregation_precision_first/04_operational_policy_freeze/FROZEN_OPERATIONAL_POLICY_FAMILY.json"
SELECTED_VALIDATION_POLICY="$REPO/reports/v4/stage1_a3/a3_11_temporal_aggregation_precision_first/01_validation_search/selected_validation_policy.json"

SELECTOR="$REPO/scripts/v4/a4a/v4_a4a_2b_validation_selector_v2.py"
SOURCE_AUDIT="$REPO/reports/v4/stage1_a4a/a4a_2b_selector_source_audit_v2"

OUT="$REPO/reports/v4/stage1_a4a/a4a_2b_validation_selection_v2"
LOG="$REPO/logs/v4/v4_a4a_2b_validation_selection_v2.log"

echo "===== V4 A4a-2B VALIDATION-ONLY SELECTION V2 ====="
echo "started_at=$(date --iso-8601=seconds)"
echo "selector=$SELECTOR"
echo "output=$OUT"
echo "development_test_access=false"
echo

python -u "$SELECTOR" \
  --mode select \
  --repo "$REPO" \
  --data-dir "$DATA" \
  --training-dir "$TRAINING_DIR" \
  --primary-trainer "$PRIMARY_TRAINER" \
  --original-trainer "$ORIGINAL_TRAINER" \
  --model-source "$MODEL_SOURCE" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --slot-module "$SLOT_MODULE" \
  --preflight-script "$PREFLIGHT_SCRIPT" \
  --preflight-dir "$PREFLIGHT_DIR" \
  --full-training-contract "$FULL_TRAINING_CONTRACT" \
  --slot-contract "$SLOT_CONTRACT" \
  --frozen-l2-policy "$FROZEN_L2_POLICY" \
  --frozen-operational-policy "$FROZEN_OPERATIONAL_POLICY" \
  --selected-validation-policy "$SELECTED_VALIDATION_POLICY" \
  --source-audit-dir "$SOURCE_AUDIT" \
  --output-dir "$OUT" \
  --batch-size 512 \
  --num-workers 2 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "a4a_2b_selection_v2_status=$status"
echo "finished_at=$(date --iso-8601=seconds)"

if [ -f "$OUT/V4_A4A_2B_VALIDATION_SELECTION_PASS" ]; then
  cat "$OUT/V4_A4A_2B_VALIDATION_SELECTION_PASS"
elif [ -f "$OUT/V4_A4A_2B_VALIDATION_SELECTION_HOLD" ]; then
  cat "$OUT/V4_A4A_2B_VALIDATION_SELECTION_HOLD"
else
  echo "NO FINAL MARKER FOUND"
fi

exit "$status"
