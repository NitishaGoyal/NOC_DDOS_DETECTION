#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/research/projects/GNN-2d}"
python "$ROOT/scripts/v5/p2/freeze_v5_p2_e0a_raw_head_structured_decoder_boundary.py" \
  --repo-root "$ROOT" \
  --e0-dir "$ROOT/reports/v5/p2_e0_validation_threshold_protocol_lock" \
  --architecture-freeze-dir "$ROOT/reports/v5/p2_task_d_architecture_selection_freeze" \
  --checkpoint-freeze-dir "$ROOT/reports/v5/p2_task_d_checkpoint_selection_freeze" \
  --output-dir "$ROOT/reports/v5/p2_e0a_raw_head_structured_decoder_boundary"
