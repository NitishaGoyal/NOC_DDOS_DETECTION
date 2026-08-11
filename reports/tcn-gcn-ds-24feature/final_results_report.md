# DS2 4x4 24-Feature Temporal GCN / TCN-GCN Results Report

## Dataset

Dataset used:

```text
data/processed/graph_dataset/paper1_temporal_graphs_ds2_4x4_portfeat.npz
```

Input tensor:

```text
x = [S, 16 routers, 8 epochs, 24 features]
```

The 24 features include aggregate router-level IFD/count features and per-port directional features. The goal is graph-level attack detection and node-level malicious-router localization.

## Models Compared

### Conv1D-GCN

Simple temporal encoder using 1D convolution, followed by a 2-layer GCN.

### TCN-GCN

Stronger temporal encoder using TCN-style residual temporal blocks, dilation, dropout, and attention pooling, followed by a 2-layer GCN.

# 1. Pre-Balanced Placement Split Results

This was the original placement split.

Important caveat:

```text
validation = 1-attacker + 2-attacker cases
test       = 3-attacker cases
```

So the split was useful, but not perfectly balanced across attacker counts.

## 1.1 Conv1D-GCN Early-Stopping Result

Model folder:

```text
models/temporal_gcn_float_placement_ds2_4x4_portfeat_es
```

Best epoch: `58`

Fixed threshold test result:

```text
graph F1            = 0.9990
node F1             = 0.9571
exact localization  = 0.7791
```

Top-k test result:

```text
top1_any            = 0.9886
topK_exact          = 0.8927
top4_all            = 0.9953
```

Interpretation: Conv1D-GCN is extremely strong for graph-level attack detection on the original placement split, and reasonably strong for localization.

## 1.2 TCN-GCN Result

Model folder:

```text
models/temporal_tcn_gcn_placement_ds2_4x4_portfeat_ep75_pat10
```

Best epoch: `14`

Fixed threshold test result:

```text
graph F1            = 0.9836
graph FPR           = 0.0000
node precision      = 0.9711
node recall         = 0.9849
node F1             = 0.9779
exact localization  = 0.9059
```

Validation-selected threshold:

```text
node threshold      = 0.30
node precision      = 0.9643
node recall         = 0.9952
node F1             = 0.9796
exact localization  = 0.8962
```

Top-k test result:

```text
top1_any            = 0.9932
topK_exact          = 0.9411
top4_all            = 0.9941
avg_worst_rank      = 3.08
```

Interpretation: On the original placement split, TCN-GCN improves localization substantially compared to Conv1D-GCN. Exact localization improves from 77.91% to 90.59%, while node F1 improves from 95.71% to 97.79%.

# 2. Balanced Placement Split Results

A balanced placement split was created to make validation and test both include:

```text
1-attacker cases
2-attacker cases
3-attacker cases
```

Split file:

```text
splits/balanced_placement_ds2_4x4_seed7.npz
```

This split is the cleaner evaluation because it tests generalization across attacker counts in both validation and test.

## 2.1 Balanced Conv1D-GCN Result

Model folder:

```text
models/temporal_gcn_float_balancedplacement_ds2_4x4_portfeat_es_seed7
```

Best epoch: `46`

Fixed threshold result from training summary:

```text
graph F1            = 0.8665
node precision      = 0.7967
node recall         = 0.8517
node F1             = 0.8232
exact localization  = 0.3976
```

Validation-selected threshold:

```text
node threshold      = 0.60
graph F1            = 0.8665
graph FPR           = 0.0222
node precision      = 0.8118
node recall         = 0.8436
node F1             = 0.8274
exact localization  = 0.4168
```

Top-k test result:

```text
top1_any            = 0.9163
topK_exact          = 0.8733
top2_all            = 0.6211
top3_all            = 0.9345
top4_all            = 0.9939
avg_worst_rank      = 2.19
```

Interpretation: Conv1D-GCN still ranks likely attackers reasonably well, but its threshold-based exact localization becomes weak on the harder balanced unseen-placement split.

## 2.2 Balanced TCN-GCN Result

Model folder:

```text
models/temporal_tcn_gcn_balancedplacement_ds2_4x4_portfeat_es_seed7
```

Best epoch: `31`

Fixed threshold test result:

```text
graph accuracy       = 0.9340
graph precision      = 0.9991
graph recall         = 0.9315
graph F1             = 0.9641
graph FPR            = 0.0162
node precision       = 0.9315
node recall          = 0.9182
node F1              = 0.9248
exact localization   = 0.7723
```

Validation-selected threshold:

```text
node threshold       = 0.55
graph F1             = 0.9641
node precision       = 0.9331
node recall          = 0.9142
node F1              = 0.9235
exact localization   = 0.7670
```

Top-k test result:

```text
top1_any             = 0.9156
topK_exact           = 0.8996
top2_all             = 0.6162
top3_all             = 0.9518
top4_all             = 0.9870
avg_worst_rank       = 2.21
```

Per-attacker-count topK_exact:

```text
1 attacker           = 0.7990
2 attackers          = 0.9517
3 attackers          = 0.9481
```

Interpretation: On the balanced unseen-placement split, TCN-GCN significantly outperforms Conv1D-GCN for both graph-level detection and node-level localization.

# 3. Final Balanced Comparison

| Model | Split | Graph F1 | Node F1 | Exact Localization | TopK Exact | Top4 All |
|---|---:|---:|---:|---:|---:|---:|
| Conv1D-GCN | balanced seed7 | 0.8665 | 0.8274 tuned | 0.4168 tuned | 0.8733 | 0.9939 |
| TCN-GCN | balanced seed7 | 0.9641 | 0.9248 fixed | 0.7723 fixed | 0.8996 | 0.9870 |

# 4. Main Conclusions

1. The 24-feature per-port directional dataset is necessary for strong attacker localization.
2. Conv1D-GCN is lightweight and performs well on easier/original placement splits.
3. On the balanced unseen-placement split, Conv1D-GCN generalization drops significantly.
4. TCN-GCN generalizes much better under the balanced split.
5. TCN-GCN is currently the best localization model.
6. Conv1D-GCN is still useful as a hardware-efficient baseline.
7. The next hardware direction should use Conv1D-GCN as the first RTL accelerator target and TCN-GCN as the accuracy-optimized reference model.

# 5. Current Best Result

For paper-quality balanced split reporting, the current best model is:

```text
TCN-GCN on balanced placement seed7
```

Main result:

```text
graph F1             = 0.9641
node F1              = 0.9248
exact localization   = 0.7723
topK exact           = 0.8996
top4 all             = 0.9870
```
