# V5 P2-G3 Role-Aware Multilabel Localization Training Package

This package implements the frozen G0 task:

```text
C_ROLE_AWARE_MULTILABEL_LOCALIZATION
```

It compares, under identical training conditions:

- Conv1D-only;
- Conv1D + GCNConv;
- Conv1D + GraphConv;
- seeds 107, 117, and 127;
- serial execution, one GPU training process at a time;
- P2 train and validation only;
- no P2 test enumeration, deserialization, or inference.

GAT is excluded because the frozen G1 and G2 decisions are both
`DO_NOT_PROMOTE_GAT`.

## Scientific variable

The only intended architecture variable is the graph operator. Every model
uses:

- the unchanged frozen B3 causal depthwise-separable Conv1D encoder;
- the unchanged B3 node projection;
- graph width 64 and two graph layers for graph models;
- ReLU after each graph layer;
- no graph dropout and no residual graph path;
- four identical-structure independent binary role heads:
  - source `[16]`;
  - transit `[16]`;
  - victim `[16]`;
  - path `[16]`.

The roles are not mutually exclusive. A router may belong to more than one
role at the same time.

The clean Conv1D path must reproduce all four frozen B3 role logits with
exactly zero maximum absolute error before scientific training is authorized.
The common temporal encoder, node projection, and four role-head initial
states must be identical across all operators in the preflight.

## Frozen G3 loss

Each role uses a separately class-weighted `BCEWithLogitsLoss`. The total loss
is the sum of the four role losses:

```text
source weighted BCE
+ transit weighted BCE
+ victim weighted BCE
+ path weighted BCE
```

Positive weights are derived only from the frozen B0-R3 P2-train label
summary, using the same `[1,20]` clamp as G1:

```text
source  = 20.0
transit = 15.7433147902343
victim  = 20.0
path    = 7.375342240922689
```

## Checkpoint selection

The best validation checkpoint is selected without threshold tuning:

```text
0.30 * source average precision
+ 0.25 * transit average precision
+ 0.25 * victim average precision
+ 0.20 * path average precision
```

## Post-checkpoint threshold selection

After the best checkpoint is frozen, four separate validation-only thresholds
are selected on the common grid:

```text
0.000 through 1.000 inclusive, step 0.001
```

Each role uses the same objective inherited from the G1 node-F1/exact-set
policy:

```text
0.60 * role node F1
+ 0.40 * exact role-set accuracy on attack items
```

Attack-item selection uses `y_attack`, so a legitimate zero-transit role set
inside an attack sample remains part of transit exact-set evaluation.

Thresholds are never tuned during training and no test data are accessed.

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

- Conv1D-only: 34,692
- GCNConv: 43,012
- GraphConv: 51,204

Unused B3 graph and attacker-count heads are absent.

## Required execution order

1. Install all package files into `scripts/v5/p2/`.
2. Run `run_v5_p2_g3_clean_training_implementation_preflight.sh`.
3. Confirm `V5_P2_G3_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_COMPLETE`.
4. Launch `run_v5_p2_g3_role_aware_multilabel_training_matrix.sh` serially.
5. The matrix automatically aggregates all nine completed runs.

## Output locations

Per-run checkpoints:

```text
models/v5/p2_g3_role_aware_multilabel/<operator>/seed_<seed>/
```

Per-run reports:

```text
reports/v5/p2_g3_role_aware_multilabel_training/<operator>/seed_<seed>/
```

Aggregation:

```text
reports/v5/p2_g3_role_aware_multilabel_matrix_aggregation/
```

The aggregation records per-seed and three-seed statistics for:

- checkpoint AP and AUROC for every role;
- role node F1, precision, recall, balanced accuracy, and FPR;
- exact role-set accuracy on attack and control samples;
- exact correctness of all four role sets jointly;
- parameter count;
- analytical operation-count proxies;
- peak CUDA memory;
- inference-latency proxy.

Metric leaders remain validation-only evidence. G3 does not select the final
architecture. G1, G2, and G3 must be reviewed jointly before the frozen full
multitask comparison.

## Security boundary

The scripts never construct a test dataset and never list or probe a P2 test
directory. Architecture, checkpoint, and threshold decisions remain
validation-only. Quantization, Legal NoC decoding, RTL generation, and blind
test evaluation are out of scope for this package.
