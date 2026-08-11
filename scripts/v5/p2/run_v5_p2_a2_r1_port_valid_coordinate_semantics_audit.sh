#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
A2="$REPO/reports/v5/p2_a2_feature_normalization_mask_contract"

SCRIPT="$REPO/scripts/v5/p2/audit_v5_p2_a2_r1_port_valid_coordinate_semantics.py"
OUT="$REPO/reports/v5/p2_a2_r1_port_valid_coordinate_semantics_audit"
LOG="$REPO/logs/v5/v5_p2_a2_r1_port_valid_coordinate_semantics_audit.log"

echo "===== V5 P2-A2-R1 PORT-VALID COORDINATE-SEMANTICS AUDIT ====="
echo "repo=$REPO"
echo "data=$DATA"
echo "historical_a2_hold=$A2"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "train_tensor_contents_accessed=sample_only"
echo "validation_tensor_contents_accessed=sample_only"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo

for path in "$DATA" "$A2"; do
    [ -d "$path" ] || {
        echo "STOP: missing directory: $path"
        exit 2
    }
done

[ -f "$SCRIPT" ] || {
    echo "STOP: missing script: $SCRIPT"
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

python -u "$SCRIPT" \
    --root "$DATA" \
    --a2-dir "$A2" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_a2_r1_status=$status"

if [ -f "$OUT/V5_P2_A2_R1_PORT_VALID_COORDINATE_SEMANTICS_AUDIT_COMPLETE" ]; then
    cat "$OUT/V5_P2_A2_R1_PORT_VALID_COORDINATE_SEMANTICS_AUDIT_COMPLETE"
elif [ -f "$OUT/V5_P2_A2_R1_PORT_VALID_COORDINATE_SEMANTICS_AUDIT_HOLD" ]; then
    cat "$OUT/V5_P2_A2_R1_PORT_VALID_COORDINATE_SEMANTICS_AUDIT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
