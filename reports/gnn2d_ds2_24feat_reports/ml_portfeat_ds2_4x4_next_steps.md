# DS2 4×4 Port-Feature Next Steps

## 1. Report both threshold settings

The uploaded notes correctly distinguish between:

```text
fixed threshold evaluation
validation-tuned threshold evaluation
```

A threshold of 0.50 is not tuning. It is fixed-threshold evaluation.

For the report, keep both:

```text
A. fixed 0.50 threshold
B. validation-tuned threshold
```

This is especially important because the placement split validation and test attacker distributions are mismatched.

## 2. Best headline result

For placement split, the fixed 0.50 result is the cleaner headline result:

```text
graph F1: 0.9990
node F1: 0.9572
exact localization: 0.7798
topK exact: 0.8934
```

## 3. Save fixed-threshold logs

Recommended log names:

```text
logs/eval_fixed050_prototype_ds2_4x4_portfeat.log
logs/eval_fixed050_placement_ds2_4x4_portfeat.log
```

## 4. Recommended fixed-threshold commands

Prototype:

```bash
python scripts/eval_temporal_gcn_fixed_threshold.py \
  --data data/processed/graph_dataset/paper1_temporal_graphs_ds2_4x4_portfeat.npz \
  --model-dir models/temporal_gcn_float_prototype_ds2_4x4_portfeat \
  --batch-size 512 \
  --graph-threshold 0.50 \
  --node-threshold 0.50 \
  | tee logs/eval_fixed050_prototype_ds2_4x4_portfeat.log
```

Placement:

```bash
python scripts/eval_temporal_gcn_fixed_threshold.py \
  --data data/processed/graph_dataset/paper1_temporal_graphs_ds2_4x4_portfeat.npz \
  --model-dir models/temporal_gcn_float_placement_ds2_4x4_portfeat \
  --batch-size 512 \
  --graph-threshold 0.50 \
  --node-threshold 0.50 \
  | tee logs/eval_fixed050_placement_ds2_4x4_portfeat.log
```

## 5. Future experiments

After reports and fixed-threshold logs are saved, useful next experiments are:

```text
1. feature ablation:
   - aggregate only
   - port counts only
   - port IFD only
   - all 24 features

2. error analysis:
   - where does localization fail?
   - which attacker placements are hardest?
   - do failures happen near victims or near paths?

3. hardware feature reduction:
   - can we reduce 24 features to 10 or 12 without losing much localization?
   - this matters for future hardware/security-engine implementation.
```
