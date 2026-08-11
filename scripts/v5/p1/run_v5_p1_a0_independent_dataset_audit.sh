#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

if [ -z "${P1_ROOT:-}" ]; then
    echo "STOP: set P1_ROOT to the P1 dataset root."
    echo
    echo "Example:"
    echo "  P1_ROOT=/absolute/path/to/p1_dataset \\"
    echo "  bash scripts/v5/p1/run_v5_p1_a0_independent_dataset_audit.sh"
    exit 2
fi

B8="$REPO/reports/v5/p0_b8_freeze_b3_conv1d_only_architecture"
SCRIPT="$REPO/scripts/v5/p1/audit_v5_p1_a0_independent_dataset.py"
OUT="$REPO/reports/v5/p1_a0_independent_dataset_audit"
LOG="$REPO/logs/v5/v5_p1_a0_independent_dataset_audit.log"

ARGS=(
    --root "$P1_ROOT"
    --b8-dir "$B8"
    --output-dir "$OUT"
)

if [ -n "${P1_TRAIN_DIR:-}" ]; then
    ARGS+=(--train-dir "$P1_TRAIN_DIR")
fi
if [ -n "${P1_VALIDATION_DIR:-}" ]; then
    ARGS+=(--validation-dir "$P1_VALIDATION_DIR")
fi
if [ -n "${P1_TEST_DIR:-}" ]; then
    ARGS+=(--test-dir "$P1_TEST_DIR")
fi

echo "===== V5 P1-A0 INDEPENDENT DATASET AUDIT ====="
echo "repo=$REPO"
echo "p1_root=$P1_ROOT"
echo "b8=$B8"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "training_performed=false"
echo "test_tensor_payloads_loaded=false"
echo

[ -d "$P1_ROOT" ] || {
    echo "STOP: P1_ROOT does not exist: $P1_ROOT"
    exit 2
}
[ -d "$B8" ] || {
    echo "STOP: B8 directory missing: $B8"
    exit 2
}
[ -f "$SCRIPT" ] || {
    echo "STOP: audit script missing: $SCRIPT"
    exit 2
}
[ ! -e "$OUT" ] || {
    echo "STOP: output already exists: $OUT"
    exit 2
}
[ ! -e "$LOG" ] || {
    echo "STOP: log already exists: $LOG"
    exit 2
}

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" "${ARGS[@]}" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

echo
echo "v5_p1_a0_status=$status"

if [ -f "$OUT/V5_P1_A0_INDEPENDENT_DATASET_AUDIT_PASS" ]; then
    cat "$OUT/V5_P1_A0_INDEPENDENT_DATASET_AUDIT_PASS"
elif [ -f "$OUT/V5_P1_A0_INDEPENDENT_DATASET_AUDIT_HOLD" ]; then
    cat "$OUT/V5_P1_A0_INDEPENDENT_DATASET_AUDIT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
