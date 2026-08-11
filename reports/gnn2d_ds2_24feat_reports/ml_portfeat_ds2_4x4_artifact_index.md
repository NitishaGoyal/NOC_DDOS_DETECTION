# DS2 4×4 Port-Feature Artifact Index

## Project directory

```text
/home/zira/research/projects/GNN-2d
```

## Dataset paths

Previous aggregate datasets:

```text
data/processed/graph_dataset/paper1_temporal_graphs_full.npz
data/processed/graph_dataset/paper1_temporal_graphs_full_4feat.npz
```

New port-feature dataset:

```text
data/processed/graph_dataset/paper1_temporal_graphs_ds2_4x4_portfeat.npz
```

Source folder for port-feature dataset:

```text
/home/zira/tools/architecture/gem5/GNN-2D-NOC/ds-2-4x4
```

## Scripts

Training:

```text
scripts/train_temporal_gcn.py
```

Validation-tuned threshold evaluation:

```text
scripts/eval_temporal_gcn_thresholds.py
```

Top-k evaluation:

```text
scripts/eval_temporal_gcn_topk.py
```

Fixed 0.50 threshold evaluation, if created:

```text
scripts/eval_temporal_gcn_fixed_threshold.py
```

Aggregate 4-feature dataset creation:

```text
scripts/add_count_features_to_graph_dataset.py
```

## Model folders

Aggregate 4-feature model:

```text
models/temporal_gcn_float_prototype_4feat
```

Port-feature prototype model:

```text
models/temporal_gcn_float_prototype_ds2_4x4_portfeat
```

Port-feature placement model:

```text
models/temporal_gcn_float_placement_ds2_4x4_portfeat
```

## Important logs

Aggregate baseline:

```text
logs/train_temporal_gcn_prototype_4feat.log
logs/eval_thresholds_prototype_4feat.log
logs/eval_topk_prototype_4feat.log
```

Port-feature prototype:

```text
logs/train_temporal_gcn_prototype_ds2_4x4_portfeat.log
logs/eval_thresholds_prototype_ds2_4x4_portfeat.log
logs/eval_topk_prototype_ds2_4x4_portfeat.log
logs/eval_fixed050_prototype_ds2_4x4_portfeat.log
```

Port-feature placement:

```text
logs/train_temporal_gcn_placement_ds2_4x4_portfeat.log
logs/eval_thresholds_placement_ds2_4x4_portfeat.log
logs/eval_topk_placement_ds2_4x4_portfeat.log
logs/eval_fixed050_placement_ds2_4x4_portfeat.log
```

## Reports to save

```text
reports/ml_portfeat_ds2_4x4_results_report.md
reports/ml_portfeat_ds2_4x4_status_summary.txt
reports/ml_portfeat_ds2_4x4_artifact_index.md
reports/ml_portfeat_ds2_4x4_next_steps.md
```

## Exact summary snapshot

```text
reports/ml_portfeat_ds2_4x4_summary_json_snapshot.md
```

This file records the exact values copied from:

```text
models/temporal_gcn_float_prototype_ds2_4x4_portfeat/summary.json
models/temporal_gcn_float_placement_ds2_4x4_portfeat/summary.json
```
