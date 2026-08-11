#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/home/zira/research/projects/GNN-2d}"
DATA_LINK="${2:-$REPO/data/processed/v5/p3_1500_d70_tranche_a_preliminary}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

E2_LOCK="$REPO/reports/v5/p3_experiments/f0_d70_feature_study/retrained_group_ablation/V5_P3_F7_E2_GENERATED_TRAINER_STATIC_AND_RUNTIME_PREFLIGHT_LOCK.json"
TRAINER=/home/zira/research/projects/GNN-2d/scripts/v5/p3/experiments/run_v5_p3_f7_reconstructed_group_ablation_trainer_e2_r5.py
ADAPTER=/home/zira/research/projects/GNN-2d/src/models/v5_p3_f7_r5_runtime_adapter.py
MATRIX_DIR=/home/zira/research/projects/GNN-2d/reports/v5/p3_experiments/f0_d70_feature_study/retrained_group_ablation/F7_E1_R5_PRIMARY_SEED107_RUN_SPECS
RUN_ROOT=/home/zira/research/projects/GNN-2d/reports/v5/p3_experiments/f0_d70_feature_study/retrained_group_ablation/f7_runs/seed107
B1_DIR=/home/zira/research/projects/GNN-2d/reports/v5/p2_b1_training_protocol_lock
B0_R3_DIR=/home/zira/research/projects/GNN-2d/reports/v5/p2_b0_r3_corrected_nontest_label_shortcut_audit
LEGACY_LOADER=/home/zira/research/projects/GNN-2d/src/data/v5_p2_pair_aligned_primary58_dataset.py
LEGACY_MODEL=/home/zira/research/projects/GNN-2d/src/models/v5_p2_b3_conv1d_only_count4.py

[[ -f "$E2_LOCK" ]] || {
  echo "F7 E2 execution lock missing: $E2_LOCK" >&2
  exit 1
}
"$PYTHON" -c 'import json,sys; from pathlib import Path; x=json.loads(Path(sys.argv[1]).read_text()); assert x.get("status")=="PASS"; assert x.get("primary_matrix_execution_authorized") is True; assert x.get("sealed_test_tensors_loaded") is False' "$E2_LOCK"

[[ -f "$TRAINER" ]] || {
  echo "generated trainer missing: $TRAINER" >&2
  exit 1
}
[[ -f "$ADAPTER" ]] || {
  echo "runtime adapter missing: $ADAPTER" >&2
  exit 1
}

mkdir -p "$RUN_ROOT"

for SPEC in "$MATRIX_DIR"/*.json; do
  LABEL="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["label"])' "$SPEC")"
  RUN_DIR="$RUN_ROOT/$LABEL"
  MODEL_DIR="$RUN_DIR/model"
  REPORT_DIR="$RUN_DIR/report"

  if [[ -f "$REPORT_DIR/F7_RUN_COMPLETE" ]]; then
    echo "skip complete run: $LABEL"
    continue
  fi

  mkdir -p "$RUN_DIR"

  if [[ -e "$MODEL_DIR" || -e "$REPORT_DIR" ]]; then
    echo "STOP: incomplete F7 run output already exists: $RUN_DIR" >&2
    echo "Remove or archive the incomplete run directory before retrying; no output will be overwritten." >&2
    exit 1
  fi

  export F7_REPO="$REPO"
  export F7_DATA_LINK="$DATA_LINK"
  export F7_RUN_DIR="$RUN_DIR"
  export F7_SEED="107"
  export F7_ABLATION_LABEL="$LABEL"

  CMD=(
    "$PYTHON"
    "$TRAINER"
    --root "$REPO"
    --b1-dir "$B1_DIR"
    --b0-r3-dir "$B0_R3_DIR"
    --loader-path "$LEGACY_LOADER"
    --model-path "$LEGACY_MODEL"
    --model-dir "$MODEL_DIR"
    --report-dir "$REPORT_DIR"
    --seed "107"
  )

  printf 'launching %s\n' "$LABEL"
  printf 'command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'

  "${CMD[@]}" 2>&1 | tee "$RUN_DIR/console.log"

  touch "$REPORT_DIR/F7_RUN_COMPLETE"
done
