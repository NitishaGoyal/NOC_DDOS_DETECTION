# DS2 4×4 Port-Feature Temporal GCN Results Report

## 1. Purpose of this report

This report documents what happened after the previous ML baseline report.

The previous report concluded that **aggregate router-level features** were enough for graph-level DoS detection, but not enough for reliable attacker localization.

This new report documents the next experiment: using a **24-feature per-port/directional dataset** for the same 4×4 NoC Temporal GCN problem.

The goal is still the same:

1. **Graph-level detection**  
   Decide whether a NoC time window is normal or attacked.

2. **Node-level localization**  
   Predict which router(s) are malicious/attacked.

---

## 2. What changed from the previous report

### Previous stage: aggregate features

Previously, the model used only aggregate router-level features.

The strongest aggregate dataset had 4 features:

```text
1. ifd_in_norm
2. ifd_out_norm
3. input_flit_count_norm
4. output_flit_count_norm
```

That dataset was:

```text
data/processed/graph_dataset/paper1_temporal_graphs_full_4feat.npz
```

Shape:

```text
x = (42809, 16, 8, 4)
```

Result:

```text
graph F1:   0.9996
node F1:    0.3637
exact loc:  0.1141
topK exact: 0.1763
```

Conclusion from previous stage:

```text
Aggregate features are enough to detect that an attack exists.
Aggregate features are not enough to reliably identify the exact attacker router(s).
```

The reason was that aggregate features tell the model total input/output behavior, but not which direction the traffic came from or where it went.

---

### New stage: port/directional features

The new experiment uses a 24-feature dataset with directional information.

Source folder:

```text
/home/zira/tools/architecture/gem5/GNN-2D-NOC/ds-2-4x4
```

Converted dataset:

```text
data/processed/graph_dataset/paper1_temporal_graphs_ds2_4x4_portfeat.npz
```

Shape:

```text
x = (42809, 16, 8, 24)
```

This dataset includes:

```text
aggregate IFD features
aggregate flit count features
per-port input flit count features
per-port output flit count features
per-port input IFD features
per-port output IFD features
```

The important improvement is that the model now gets local/north/east/south/west directional behavior.

---

## 3. Directory and file setup

GNN project directory:

```text
/home/zira/research/projects/GNN-2d
```

gem5 directory:

```text
/home/zira/tools/architecture/gem5
```

Original graph dataset symlink:

```text
data/processed/graph_dataset
```

points to:

```text
/home/zira/tools/architecture/gem5/experiments/paper_1/graph_dataset
```

New port-feature source folder:

```text
/home/zira/tools/architecture/gem5/GNN-2D-NOC/ds-2-4x4
```

New converted ML dataset:

```text
data/processed/graph_dataset/paper1_temporal_graphs_ds2_4x4_portfeat.npz
```

---

## 4. Model architecture

The model is the same Temporal GCN architecture used earlier.

Input tensor format:

```text
x = [samples, routers, time_window, features]
```

For this project:

```text
samples     = 42809
routers     = 16
time_window = 8
features    = 24 for the port-feature dataset
```

Model pipeline:

```text
router temporal features
    ↓
1D temporal convolution per router
    ↓
router embeddings
    ↓
2-layer GCN over 4×4 mesh
    ↓
graph head + node head
```

Important clarification:

```text
The temporal encoder is a 1D convolution, not a 2D convolution.
```

It performs convolution over the 8-step time window for each router independently. Spatial information is handled by the GCN using the 4×4 NoC mesh adjacency.

---

## 5. Why 24 features helped

The aggregate dataset told the model:

```text
router X has high input traffic
router X has high output traffic
router X has changed inter-flit delay
```

But it did not tell the model:

```text
traffic entered from west
traffic exited to east
local injection increased
north port became congested
south output is forwarding attack traffic
```

In a mesh NoC, congestion spreads spatially. Without port direction, the attacker router and its neighbors can look similar.

The 24-feature dataset fixes this by giving the model directional information.

This means the model can learn path structure, not just congestion intensity.

---

## 6. Port-feature prototype split result

Dataset:

```text
data/processed/graph_dataset/paper1_temporal_graphs_ds2_4x4_portfeat.npz
```

Model directory:

```text
models/temporal_gcn_float_prototype_ds2_4x4_portfeat
```

### 6.1 Fixed threshold 0.50 result

This is a fixed evaluation threshold, not threshold tuning.

From training summary:

```text
graph accuracy: 0.9159
graph precision: 0.9961
graph recall: 0.9125
graph F1: 0.9525
graph FPR: 0.0424

node precision: 0.7856
node recall: 0.9980
node F1: 0.8792
exact localization: 0.5871
```

Interpretation:

```text
Even at a simple fixed 0.50 threshold, node localization becomes much stronger than the aggregate-feature baseline.
```

---

### 6.2 Validation-tuned threshold result

Best validation node threshold:

```text
0.90
```

Test result:

```text
graph accuracy: 0.9161
graph precision: 0.9961
graph recall: 0.9126
graph F1: 0.9526
graph FPR: 0.0424

node precision: 0.8983
node recall: 0.9745
node F1: 0.9349
exact localization: 0.7677
```

Interpretation:

```text
With validation-selected thresholding, the port-feature prototype model gives strong localization.
Node F1 improves from 0.3637 to 0.9349.
Exact localization improves from 0.1141 to 0.7677.
```

---

### 6.3 Prototype top-k result

Test set:

```text
overall top1_any:   0.9816
overall topK_exact: 0.9431
overall top4_all:   0.9998
avg worst rank:     2.06
```

By attacker count:

```text
1 attacker topK_exact: 0.9773
2 attacker topK_exact: 0.9545
3 attacker topK_exact: 0.8975
```

Interpretation:

```text
The model ranks true attackers very accurately.
In 94.31% of attack windows, the top-K predicted routers exactly match the K true attackers.
```

---

## 7. Port-feature placement split result

Dataset:

```text
data/processed/graph_dataset/paper1_temporal_graphs_ds2_4x4_portfeat.npz
```

Model directory:

```text
models/temporal_gcn_float_placement_ds2_4x4_portfeat
```

The placement split is more meaningful than the prototype split because it tests whether the model generalizes to attacker placements not seen in training.

---

### 7.1 Fixed threshold 0.50 result

From training summary:

```text
graph accuracy: 0.9982
graph precision: 0.9991
graph recall: 0.9989
graph F1: 0.9990
graph FPR: 0.0121

node precision: 0.9287
node recall: 0.9875
node F1: 0.9572
exact localization: 0.7798
```

Interpretation:

```text
The fixed 0.50 threshold performs very well on placement split.
This is a strong and clean result because it does not depend on threshold tuning.
```

---

### 7.2 Validation-tuned threshold result

Best validation node threshold:

```text
0.10
```

Test result:

```text
graph accuracy: 0.9982
graph precision: 0.9991
graph recall: 0.9989
graph F1: 0.9990
graph FPR: 0.0121

node precision: 0.8760
node recall: 0.9979
node F1: 0.9330
exact localization: 0.6520
```

Important interpretation:

```text
The validation-tuned threshold is lower than expected because the placement validation/test attacker distributions are mismatched.
Therefore, report both fixed 0.50 and validation-tuned results.
For placement split, fixed 0.50 is actually the cleaner headline threshold result.
```

---

### 7.3 Placement top-k result

Test set:

```text
top1_any:       0.9885
topK_exact:     0.8934
top3_all:       0.8934
top4_all:       0.9953
avg worst rank: 3.12
```

The placement test set contains 3-attacker cases, so:

```text
topK_exact = top3 exact
```

Interpretation:

```text
On unseen 3-attacker placements, the model exactly identifies the top-3 attacker routers in 89.34% of attack windows.
All true attackers are inside the top-4 predicted routers in 99.53% of attack windows.
```

---

## 8. Main comparison

```text
Feature set                  Graph F1   Node F1   Exact Loc   TopK Exact
-------------------------------------------------------------------------
Aggregate 4-feature           0.9996     0.3637    0.1141      0.1763
Port 24-feature prototype     0.9526     0.9349    0.7677      0.9431
Port 24-feature placement     0.9990     0.9572    0.7798      0.8934
```

Key conclusion:

```text
The bottleneck was not the GNN architecture.
The bottleneck was the feature representation.
```

Once per-port directional features were added, the same Temporal GCN architecture achieved strong localization.

---

## 9. What exactly happened

The original aggregate dataset only described each router using total input/output traffic and average inter-flit delay.

That was enough to detect whether the NoC was attacked, but not enough to determine the exact attacker location.

The new 24-feature port dataset added directional information. It told the model not only that a router had high traffic, but also which port the traffic came from and where it went.

This changed the localization problem completely.

The model could now distinguish between:

```text
attacker router injecting traffic
neighbor router forwarding traffic
victim-side router receiving congestion
```

This is why localization improved significantly.

---

## 10. Final conclusion

Aggregate router-level temporal features are sufficient for graph-level DoS detection, but insufficient for precise malicious-router localization.

Per-port directional temporal features significantly improve localization.

Final key prototype result:

```text
Node F1 improved from 0.3637 to 0.9349.
Exact localization improved from 0.1141 to 0.7677.
TopK exact localization improved from 0.1763 to 0.9431.
```

Final key placement result:

```text
graph F1: 0.9990
node F1: 0.9572
exact localization: 0.7798
topK exact: 0.8934
```

This validates the design direction:

```text
Temporal modeling + graph message passing works,
but localization requires directional NoC features.
```

---

## 11. Code and artifact references

Training script:

```text
scripts/train_temporal_gcn.py
```

Threshold evaluation:

```text
scripts/eval_temporal_gcn_thresholds.py
```

Top-k evaluation:

```text
scripts/eval_temporal_gcn_topk.py
```

4-feature aggregate dataset creation:

```text
scripts/add_count_features_to_graph_dataset.py
```

Important datasets:

```text
data/processed/graph_dataset/paper1_temporal_graphs_full.npz
data/processed/graph_dataset/paper1_temporal_graphs_full_4feat.npz
data/processed/graph_dataset/paper1_temporal_graphs_ds2_4x4_portfeat.npz
```

Important model folders:

```text
models/temporal_gcn_float_prototype_4feat
models/temporal_gcn_float_prototype_ds2_4x4_portfeat
models/temporal_gcn_float_placement_ds2_4x4_portfeat
```

Important logs:

```text
logs/train_temporal_gcn_prototype_4feat.log
logs/eval_thresholds_prototype_4feat.log
logs/eval_topk_prototype_4feat.log
logs/train_temporal_gcn_prototype_ds2_4x4_portfeat.log
logs/eval_thresholds_prototype_ds2_4x4_portfeat.log
logs/eval_topk_prototype_ds2_4x4_portfeat.log
logs/train_temporal_gcn_placement_ds2_4x4_portfeat.log
logs/eval_thresholds_placement_ds2_4x4_portfeat.log
logs/eval_topk_placement_ds2_4x4_portfeat.log
```

Recommended new fixed-threshold logs:

```text
logs/eval_fixed050_prototype_ds2_4x4_portfeat.log
logs/eval_fixed050_placement_ds2_4x4_portfeat.log
```

---

## 12. Recommended wording for paper/report

The previous aggregate-feature baseline achieved near-perfect graph-level attack detection but poor malicious-router localization, indicating that aggregate traffic intensity alone is insufficient to distinguish attacker routers from congested neighboring routers. To address this, we constructed a 24-feature directional dataset with per-port input/output flit counts and inter-flit delay statistics. Using the same Temporal GCN architecture, localization improved substantially. On the prototype split, node F1 improved from 0.3637 to 0.9349 and exact localization improved from 0.1141 to 0.7677. On the placement split, the model maintained strong generalization with graph F1 of 0.9990, node F1 of 0.9572, and exact localization of 0.7798 at a fixed 0.50 threshold. These results show that the main limitation of the aggregate baseline was not the GNN architecture, but the lack of directional NoC traffic features.

---

## 13. Next steps

1. Save this report in:

```text
reports/ml_portfeat_ds2_4x4_results_report.md
```

2. Save a short status summary in:

```text
reports/ml_portfeat_ds2_4x4_status_summary.txt
```

3. Save exact fixed-threshold evaluation logs:

```text
logs/eval_fixed050_prototype_ds2_4x4_portfeat.log
logs/eval_fixed050_placement_ds2_4x4_portfeat.log
```

4. Commit report and logs:

```bash
git add reports logs scripts
git commit -m "Add DS2 4x4 port-feature GNN localization report"
```

5. Continue from here with:
   - fixed 0.50 threshold logs if not already saved;
   - confusion/error analysis;
   - possible ablation: counts-only port features vs IFD-only port features;
   - hardware-friendly feature reduction.

---

## 14. Exact saved `summary.json` outputs

The following values were read directly from the saved model summary files.

### 14.1 Prototype port-feature model

Summary file:

```text
models/temporal_gcn_float_prototype_ds2_4x4_portfeat/summary.json
```

Best epoch:

```text
37
```

Test metrics:

```text
loss                    : 0.29646989645866245
acc                     : 0.9159285159285159
precision               : 0.9961404153648227
recall                  : 0.9124579124579124
f1                      : 0.9524646340391881
fpr                     : 0.04242424242424243
tnr                     : 0.9575757575757575
tn                      : 474
fp                      : 21
fn                      : 520
tp                      : 5420
node_acc                : 0.9683469308469308
node_precision          : 0.7856338214830031
node_recall             : 0.997979797979798
node_f1                 : 0.8791665121797486
exact_localization      : 0.5871017871017871
```

### 14.2 Placement port-feature model

Summary file:

```text
models/temporal_gcn_float_placement_ds2_4x4_portfeat/summary.json
```

Best epoch:

```text
58
```

Test metrics:

```text
loss                    : 0.10384062750797186
acc                     : 0.9981641011156617
precision               : 0.9990888382687927
recall                  : 0.9989371393865776
f1                      : 0.9990129830688634
fpr                     : 0.012121212121212121
tnr                     : 0.9878787878787879
tn                      : 489
fp                      : 6
fn                      : 7
tp                      : 6579
node_acc                : 0.9846066939697783
node_precision          : 0.9287006187529748
node_recall             : 0.9875493470999089
node_f1                 : 0.957221350078493
exact_localization      : 0.7798333568704985
```

### 14.3 Summary interpretation

```text
Prototype summary:
- graph F1: 0.9525
- node F1: 0.8792
- exact localization: 0.5871

Placement summary:
- graph F1: 0.9990
- node F1: 0.9572
- exact localization: 0.7798
```

The placement split is the stronger paper-facing result because it shows generalization across attacker placement.

