#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

ONE_SEED="$REPO/scripts/v5/p2/run_v5_p2_b2_one_seed.sh"
FINALIZE="$REPO/scripts/v5/p2/run_v5_p2_b2_finalize_multi_seed.sh"

for path in "$ONE_SEED" "$FINALIZE"; do
    [ -f "$path" ] || {
        echo "STOP: missing launcher: $path"
        exit 2
    }
done

for seed in 107 117 127; do
    echo
    echo "######## STARTING P2-B2 SEED $seed ########"
    bash "$ONE_SEED" "$seed" || exit $?
    echo "######## COMPLETED P2-B2 SEED $seed ########"
done

echo
echo "######## FINALIZING P2-B2 MULTI-SEED RUN ########"
bash "$FINALIZE"
