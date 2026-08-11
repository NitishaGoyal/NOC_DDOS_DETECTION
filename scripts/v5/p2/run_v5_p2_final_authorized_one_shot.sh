#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zira/research/projects/GNN-2d"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3)"; fi
EVALUATOR="/home/zira/research/projects/GNN-2d/scripts/v5/p2/evaluate_v5_p2_final_raw_a0_a1_one_shot.py"
DATA_ROOT="/home/zira/research/projects/GNN-2d/data/processed/v5/p2_complete558_dynamic_graph"
OUTPUT_DIR="/home/zira/research/projects/GNN-2d/reports/v5/p2_final_raw_a0_a1_one_shot_blind_evaluation"
AUTHORIZATION_CONTRACT="/home/zira/research/projects/GNN-2d/artifacts/v5/p2_final_no_data_preflight_and_authorization/V5_P2_FINAL_ONE_SHOT_AUTHORIZATION.json"
AUTHORIZATION_TOKEN="/home/zira/research/projects/GNN-2d/artifacts/v5/p2_final_no_data_preflight_and_authorization/V5_P2_FINAL_ONE_SHOT_AUTHORIZATION_TOKEN.json"
EXPECTED_EVALUATOR_SHA256="f6f061177b1621e3e7d8b3cbb32a401f83cdbf0488235125c62f3be17501ee44"

observed_sha="$(sha256sum "$EVALUATOR" | awk '{print $1}')"
if [[ "$observed_sha" != "$EXPECTED_EVALUATOR_SHA256" ]]; then echo "STOP: evaluator SHA-256 changed" >&2; exit 2; fi
if [[ ! -f "$AUTHORIZATION_TOKEN" ]]; then echo "STOP: authorization token is absent or already consumed" >&2; exit 2; fi
if [[ -e "${AUTHORIZATION_TOKEN}.consumed" ]]; then echo "STOP: authorization token was already consumed" >&2; exit 2; fi
if [[ -e "$OUTPUT_DIR" ]]; then echo "STOP: one-shot output guard already exists: $OUTPUT_DIR" >&2; exit 2; fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "$PYTHON" "$EVALUATOR" \
  --repo-root "$ROOT" \
  --data-root "$DATA_ROOT" \
  --authorization-contract "$AUTHORIZATION_CONTRACT" \
  --authorization-token "$AUTHORIZATION_TOKEN" \
  --output-dir "$OUTPUT_DIR"
