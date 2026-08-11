# Stage 9 B1 implementation

## Exact intervention

B1 changes only the optimization sampling distribution:

```text
class → uniform run within class → uniform window within run
```

with replacement for exactly 148,185 draws.

```text
P(normal sample) = 0.5 / (14 × 3293)
P(attack sample) = 0.5 / (31 × 3293)
normal/attack per-window weight ratio = 31/14
```

Equivalent relative weights are normal `31`, attack `14`.

Expected draws:

```text
normal class ≈ 74,092.5
attack class ≈ 74,092.5
each normal run ≈ 5,292.32
each attack run ≈ 2,389.96
```

The implementation uses `torch.multinomial`, matching
`WeightedRandomSampler` semantics, with a small audit wrapper to preserve exact
local/global sampled indices and generator-state hashes.

## Train metrics

- balanced `train_loader`: optimization only;
- deterministic full `train_eval_loader`: reporting only;
- validation and test loaders: unchanged.

This reporting correction does not affect gradients, optimizer updates,
checkpoint selection, validation, or test.

## Build

From the repository root:

```bash
cd ~/research/projects/GNN-2d
source .venv/bin/activate

sha256sum scripts/train_temporal_gcn_v3.py
```

Require:

```text
2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b
```

Then:

```bash
python /path/to/build_stage9_b1_script.py

python -m py_compile \
  scripts/train_temporal_gcn_v3_stage9_b1.py
```

Freeze and inspect the diff:

```bash
mkdir -p \
  reports/v3_failure_analysis/stage9_round1/b1_scenario_balanced/freeze

git diff --no-index -- \
  scripts/train_temporal_gcn_v3.py \
  scripts/train_temporal_gcn_v3_stage9_b1.py \
  | tee \
  reports/v3_failure_analysis/stage9_round1/b1_scenario_balanced/freeze/b1_source_diff.patch
```

## Dry run only

```bash
python scripts/train_temporal_gcn_v3_stage9_b1.py \
  --data "$HOME/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3" \
  --out-dir models/v3/stage9_b1_conv1d_class_run_balanced_seed7 \
  --split-mode v3 \
  --epochs 100 \
  --patience 15 \
  --min-delta 1e-4 \
  --batch-size 256 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --temporal-dim 8 \
  --gcn-hidden 16 \
  --gcn-out 8 \
  --node-loss-weight 1.0 \
  --graph-threshold 0.5 \
  --node-threshold 0.5 \
  --seed 7 \
  --sampler-mode class_run_balanced \
  --sampler-seed 7001 \
  --samples-per-epoch 148185 \
  --sampler-log-dir reports/v3_failure_analysis/stage9_round1/b1_scenario_balanced/dry_run \
  --dry-run-sampler
```

The dry run must finish with:

```text
verdict: PASS
batches: 579
parameter count: 882
training started: False
checkpoint created: False
```

Inspect:

```bash
cat \
  reports/v3_failure_analysis/stage9_round1/b1_scenario_balanced/dry_run/b1_dry_run_validation.txt

python -m json.tool \
  reports/v3_failure_analysis/stage9_round1/b1_scenario_balanced/dry_run/b1_dry_run_summary.json
```

Do not run GPU training until the dry-run verdict and frozen source diff pass.
