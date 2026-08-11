# V5 P2-G2 Direct Graph-Detection Training Package

This package implements the frozen G0 task:

```text
B_DIRECT_GRAPH_DETECTION
```

It compares, under identical training and readout conditions:

- Conv1D-only;
- Conv1D + GCNConv;
- Conv1D + GraphConv;
- Conv1D + GATConv screening;
- seeds 107, 117, and 127;
- serial execution, one GPU training process at a time;
- P2 train and validation only;
- no P2 test enumeration, deserialization, or inference.

## Scientific variable

The only intended architecture variable is the graph operator. Every model uses:

- the unchanged frozen B3 causal depthwise-separable Conv1D encoder;
- the unchanged B3 node projection;
- graph width 64 and two graph layers for graph models;
- ReLU after each graph layer;
- no dropout and no residual graph path;
- the same readout across every operator:
  - node mean pooling;
  - node max pooling;
  - concatenation to 128 dimensions;
  - frozen-structure `Linear(128,64) + ReLU` graph projection;
  - `Linear(64,1)` direct attack head.

The clean Conv1D model is required to reproduce the frozen B3 attack logit with
zero maximum absolute error before training. The common encoder and graph-head
initial states are also required to match across all four operators in the
preflight.

## Frozen G2 objective

Loss:

```text
unweighted BCEWithLogitsLoss(direct_graph_logit, y_attack)
```

The loss is intentionally not source-derived and does not use source, count,
transit, victim, or path targets.

Checkpoint score:

```text
0.50 * graph average precision
+ 0.50 * graph AUROC
```

Post-checkpoint threshold selection:

```text
maximize validation balanced accuracy
```

A deterministic common threshold grid is used:

```text
0.000 through 1.000 inclusive, step 0.001
```

Threshold tuning occurs only after the best validation checkpoint has been
selected. It is never performed during training.

## Training settings

- item batch size: 256, represented as 128 aligned ATTACK/CONTROL pair blocks;
- optimizer: AdamW;
- learning rate: 0.001;
- weight decay: 0.0001;
- scheduler: ReduceLROnPlateau, mode=max, factor=0.5, patience=4, min LR=1e-5;
- maximum epochs: 100;
- minimum epoch before stopping: 15;
- early-stop patience: 12;
- early-stop minimum score delta: 1e-4;
- global gradient clipping: 1.0;
- AMP: disabled;
- deterministic algorithms: enabled with warning-only fallback.

## Clean production parameter counts

- Conv1D-only: 34,561
- GCNConv: 42,881
- GraphConv: 51,073
- GATConv: 43,137

Unused B3 source/count/transit/victim/path heads are absent.

## Required execution order

1. Install all package files into `scripts/v5/p2/`.
2. Run `run_v5_p2_g2_clean_training_implementation_preflight.sh`.
3. Confirm `V5_P2_G2_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_COMPLETE`.
4. Launch `run_v5_p2_g2_direct_graph_training_matrix.sh` serially.
5. The matrix automatically aggregates all twelve completed runs.

## Output locations

Per-run checkpoints:

```text
models/v5/p2_g2_direct_graph/<operator>/seed_<seed>/
```

Per-run reports:

```text
reports/v5/p2_g2_direct_graph_training/<operator>/seed_<seed>/
```

Aggregation:

```text
reports/v5/p2_g2_direct_graph_matrix_aggregation/
```

The aggregation reports per-seed metrics, mean and sample standard deviation,
parameter count, operation-count proxies, peak CUDA memory, latency proxy, and
the frozen GAT-vs-GraphConv screening gates. It does not select the final RTL
architecture.

## Security boundary

The scripts never construct a test dataset and never list or probe a P2 test
directory. All architecture, checkpoint, and threshold decisions remain
validation-only. Quantization, RTL generation, and blind-test evaluation remain
out of scope for G2.
