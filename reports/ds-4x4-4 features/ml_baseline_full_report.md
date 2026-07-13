# ML Baseline Full Report
## Temporal GCN for 4×4 NoC DoS Detection and Localization

## 1. What this report documents

This report records the current ML baseline for the 2D NoC GNN project.

The goal was to train a temporal graph neural network on gem5/Garnet NoC traffic data for two tasks:

1. Graph-level attack detection: detect whether a NoC time window is normal or attacked.
2. Node-level attacker localization: identify which router(s) are malicious/attacked.

The current NoC topology is a 4×4 2D mesh with 16 routers.

---

## 2. Final conclusion

The current aggregate router-level dataset is good enough for graph-level attack detection, but not good enough for reliable malicious-router localization.

Best 4-feature prototype result:

```text
graph accuracy: 0.9992
graph F1:       0.9996
graph FPR:      0.0061

node precision: 0.2366
node recall:    0.7857
node F1:        0.3637
exact loc:      0.1141
```

Top-k localization:

```text
1 attacker top-k exact: 0.4970
2 attacker top-k exact: 0.0313
3 attacker top-k exact: 0.0005
```

Meaning:

```text
Attack detection works very well.
Exact attacker localization is still weak, especially for multiple attackers.
```

---

## 3. Directory setup

gem5 path:

```text
/home/zira/tools/architecture/gem5
```

Paper 1 gem5 experiment path:

```text
/home/zira/tools/architecture/gem5/experiments/paper_1
```

GNN project path:

```text
/home/zira/research/projects/GNN-2d
```

Important GNN folders:

```text
scripts/   -> training and evaluation scripts
logs/      -> saved terminal logs
models/    -> trained models and summary JSONs
reports/   -> documentation
data/      -> symlinked dataset access
```

---

## 4. Dataset linking

The dataset was not copied into the GNN project. It was linked.

Symlink inside GNN project:

```text
/home/zira/research/projects/GNN-2d/data/processed/graph_dataset
```

Points to:

```text
/home/zira/tools/architecture/gem5/experiments/paper_1/graph_dataset
```

Command used:

```bash
ln -sfn /home/zira/tools/architecture/gem5/experiments/paper_1/graph_dataset data/processed/graph_dataset
```

Conclusion:

```text
The symlink works.
The dataset being generated on another PC was not the issue.
The .npz files load correctly and training works.
```

---

## 5. Original dataset verification

Original dataset:

```text
data/processed/graph_dataset/paper1_temporal_graphs_full.npz
```

Observed shape:

```text
x                    shape=(42809, 16, 8, 2) dtype=float32
y_graph              shape=(42809,) dtype=float32
y_node               shape=(42809, 16) dtype=float32
edge_index           shape=(2, 64) dtype=int64
run_id               shape=(42809,) dtype=<U23
end_epoch            shape=(42809,) dtype=int32
attackers            shape=(42809,) dtype=<U7
flit_count_window    shape=(42809, 16, 8, 2) dtype=int32
```

Meaning:

```text
42809 = number of temporal graph samples
16    = routers in 4×4 mesh
8     = temporal window length
2     = features per router per time step
```

Original 2 features:

```text
x[..., 0] = ifd_in_norm
x[..., 1] = ifd_out_norm
```

---

## 6. Graph structure

The dataset has:

```text
edge_index shape = (2, 64)
```

For a 4×4 mesh this is consistent with:

```text
48 directed physical mesh edges + 16 self-loops = 64 edges
```

So the graph topology is valid.

---

## 7. Raw epoch CSV verification

Epoch feature CSVs were found at:

```text
/home/zira/tools/architecture/gem5/experiments/paper_1/epoch_features_full
```

CSV header:

```text
run_id
label
epoch_id
router_id
input_gap_sum
input_gap_count
output_gap_sum
output_gap_count
ifd_in
ifd_out
ifd_in_norm
ifd_out_norm
input_flit_count
output_flit_count
```

Available useful features:

```text
ifd_in_norm
ifd_out_norm
input_flit_count
output_flit_count
```

Missing directional features:

```text
input_flit_count_local/north/east/south/west
output_flit_count_local/north/east/south/west
ifd_in_local/north/east/south/west
ifd_out_local/north/east/south/west
```

Conclusion:

```text
The current saved CSVs contain aggregate router-level features only.
They do not contain per-port directional features.
```

---

## 8. Raw trace check

A search for raw trace files found no usable trace files.

The traces directory is empty.

Conclusion:

```text
The ML side currently has epoch-level aggregate files and graph datasets.
It does not have raw per-flit/per-event traces.
Therefore, directional/per-port features cannot be reconstructed from the current saved data.
```

---

## 9. Fake 4×4 GNN sanity check

Before using the real gem5 dataset, a fake 4×4 GNN was created.

Purpose:

```text
Validate that:
- 4×4 mesh graph construction works
- 1-hop/2-hop labels make sense
- 2-layer message passing learns spatial node labels
```

Fake result:

```text
accuracy: 0.94
normal F1: 0.95
second-hop F1: 0.88
first-hop/attacked F1: 0.97
```

Conclusion:

```text
The GNN/message-passing pipeline works on a controlled toy example.
```

---

## 10. 4-feature dataset creation

Original dataset:

```text
x shape = (42809, 16, 8, 2)
```

New 4-feature dataset:

```text
data/processed/graph_dataset/paper1_temporal_graphs_full_4feat.npz
```

New shape:

```text
x shape = (42809, 16, 8, 4)
```

Feature names:

```text
1. ifd_in_norm
2. ifd_out_norm
3. input_flit_count_norm
4. output_flit_count_norm
```

Script:

```text
scripts/add_count_features_to_graph_dataset.py
```

Command used:

```bash
python scripts/add_count_features_to_graph_dataset.py \
  --in-data data/processed/graph_dataset/paper1_temporal_graphs_full.npz \
  --out-data data/processed/graph_dataset/paper1_temporal_graphs_full_4feat.npz \
  --count-clip 256
```

---

## 11. Count normalization

Flit count distribution:

```text
min: 0
max: 353
p50: 74
p75: 101
p90: 127
p95: 143
p99: 174
p99.5: 186
p99.9: 218
p100: 353
```

Chosen clipping value:

```text
count_clip = 256
```

Only about this much was clipped:

```text
0.0248%
```

Normalization:

```text
count_norm = clip(count / 256, 0, 1)
```

---

## 12. Main training script

Main script:

```text
scripts/train_temporal_gcn.py
```

Important components:

```text
NoCTemporalGraphDataset
make_splits
build_normalized_adjacency
TemporalEncoder
GCNLayer
TemporalGCN
evaluate
training loop
checkpoint saving
summary writing
```

Model architecture:

```text
Input: [B, 16, 8, F]

TemporalEncoder:
- shared Conv1D over 8 time steps for each router
- produces router embedding

GCN:
- 2 graph convolution layers over the 4×4 mesh

Heads:
- graph_head for normal/attack detection
- node_head for malicious-router localization
```

High-level flow:

```text
router temporal features
        ↓
1D temporal encoder
        ↓
router embeddings
        ↓
2-layer GCN
        ↓
graph prediction + node prediction
```

Important patch:

```text
The training script was patched to use dynamic input feature dimension.
```

This allows:

```text
2-feature dataset
4-feature dataset
future port-feature dataset
```

---

## 13. PyTorch checkpoint issue

A checkpoint loading issue happened after training due to newer PyTorch behavior.

Error type:

```text
Weights only load failed
```

Fix:

```python
torch.load(..., weights_only=False)
```

This is safe because the checkpoint was locally generated by the same script.

---

## 14. Evaluation scripts

Threshold evaluation script:

```text
scripts/eval_temporal_gcn_thresholds.py
```

Purpose:

```text
Sweep node thresholds on validation set.
Choose best validation threshold by node F1.
Evaluate test set using chosen threshold.
```

Top-k evaluation script:

```text
scripts/eval_temporal_gcn_topk.py
```

Purpose:

```text
Check whether true attackers are ranked near the top by node probability.
```

Metrics:

```text
top1_any
topK_exact
top2_all
top3_all
top4_all
avg_worst_rank
```

---

## 15. Split modes

The training script supports:

```text
prototype
placement
```

Prototype split:

```text
first 70% -> train
next 15%  -> validation
last 15%  -> test
```

Purpose:

```text
Sanity-check split.
Good for proving the model can learn.
Not final paper-quality generalization.
```

Placement split:

```text
Attack runs are split by attacker placement.
This tests generalization to unseen attacker placements.
```

Current completed results are prototype results.

---

## 16. 2-feature prototype result

Dataset:

```text
data/processed/graph_dataset/paper1_temporal_graphs_full.npz
```

Model directory:

```text
models/temporal_gcn_float_prototype
```

Feature set:

```text
ifd_in_norm
ifd_out_norm
```

Best validation threshold:

```text
0.70
```

Test metrics:

```text
graph accuracy:   0.9677
graph precision:  0.9972
graph recall:     0.9677
graph F1:         0.9822
graph FPR:        0.0323

node precision:   0.2470
node recall:      0.8158
node F1:          0.3792
exact localization: 0.0828
```

Conclusion:

```text
2-feature model detects attack windows well.
Localization remains weak.
```

---

## 17. 4-feature prototype result

Dataset:

```text
data/processed/graph_dataset/paper1_temporal_graphs_full_4feat.npz
```

Model directory:

```text
models/temporal_gcn_float_prototype_4feat
```

Feature set:

```text
ifd_in_norm
ifd_out_norm
input_flit_count_norm
output_flit_count_norm
```

Training result:

```text
best epoch: 40

test graph accuracy:  0.9992
test graph precision: 0.9995
test graph recall:    0.9997
test graph F1:        0.9996
test graph FPR:       0.0061

test node precision:  0.2241
test node recall:     0.9491
test node F1:         0.3625
exact localization:   0.0948
```

Threshold sweep selected:

```text
best validation node threshold: 0.60
```

Test result at threshold 0.60:

```text
graph accuracy:   0.9992
graph precision:  0.9995
graph recall:     0.9997
graph F1:         0.9996
graph FPR:        0.0061

node precision:   0.2366
node recall:      0.7857
node F1:          0.3637
exact localization: 0.1141
```

Conclusion:

```text
Adding flit counts makes graph-level attack detection nearly perfect.
It improves exact localization slightly.
Node localization is still weak.
```

---

## 18. Top-k localization result

For 4-feature prototype:

```text
logs/eval_topk_prototype_4feat.log
```

Overall test result:

```text
top1_any:        0.2811
topK_exact:      0.1763
top2_all:        0.2451
top3_all:        0.3093
top4_all:        0.3598
avg_worst_rank:  6.47
```

By attacker count:

```text
1 attacker:
top1_any:        0.4970
topK_exact:      0.4970
top4_all:        0.8758
avg_worst_rank:  2.28

2 attackers:
top1_any:        0.1879
topK_exact:      0.0313
top4_all:        0.2015
avg_worst_rank:  6.52

3 attackers:
top1_any:        0.1586
topK_exact:      0.0005
top4_all:        0.0020
avg_worst_rank:  10.59
```

Interpretation:

```text
Single-attacker candidate ranking is somewhat useful.
Multi-attacker exact localization is poor.
```

---

## 19. Comparison table

```text
Experiment              Graph F1   Node F1   Exact Loc
-------------------------------------------------------
IFD only                0.9822     0.3792    0.0828
IFD + flit counts       0.9996     0.3637    0.1141
```

Interpretation:

```text
4-feature model is better for graph detection.
4-feature model slightly improves exact localization.
Neither feature set gives reliable attacker localization.
```

---

## 20. Why localization is weak

Current aggregate features tell the model:

```text
router X has high input traffic
router X has high output traffic
router X has changed IFD
```

They do not tell the model:

```text
traffic entered from west
traffic exited to east
local injection increased
north port is congested
south output is forwarding attack traffic
```

In a mesh NoC, congestion spreads spatially. Neighbor routers can look suspicious even when they are not attackers.

That causes:

```text
high node recall
low node precision
low exact localization
```

So this is a feature limitation, not a broken model.

---

## 21. What should happen next

Do not keep optimizing localization on the current aggregate dataset.

Next step should be frontend/gem5 feature generation.

Need per-port directional features:

```text
input_flit_count_local/north/east/south/west
output_flit_count_local/north/east/south/west
```

Ideally also:

```text
ifd_in_local/north/east/south/west
ifd_out_local/north/east/south/west
```

Then build:

```text
paper1_temporal_graphs_full_portfeat.npz
```

Expected shape options:

```text
[S, 16, 8, 12]
```

for:

```text
2 aggregate IFD + 10 per-port count features
```

or:

```text
[S, 16, 8, 22]
```

for:

```text
2 aggregate IFD + 10 per-port count + 10 per-port IFD
```

---

## 22. File reference index

Scripts:

```text
scripts/train_temporal_gcn.py
scripts/eval_temporal_gcn_thresholds.py
scripts/eval_temporal_gcn_topk.py
scripts/add_count_features_to_graph_dataset.py
```

Datasets:

```text
data/processed/graph_dataset/paper1_temporal_graphs_full.npz
data/processed/graph_dataset/paper1_temporal_graphs_full_4feat.npz
```

2-feature model artifacts:

```text
models/temporal_gcn_float_prototype/best_model.pt
models/temporal_gcn_float_prototype/history.json
models/temporal_gcn_float_prototype/summary.json
models/temporal_gcn_float_prototype/splits.npz
```

4-feature model artifacts:

```text
models/temporal_gcn_float_prototype_4feat/best_model.pt
models/temporal_gcn_float_prototype_4feat/history.json
models/temporal_gcn_float_prototype_4feat/summary.json
models/temporal_gcn_float_prototype_4feat/splits.npz
```

Logs:

```text
logs/train_temporal_gcn_prototype.log
logs/eval_thresholds_prototype.log
logs/train_temporal_gcn_prototype_4feat.log
logs/eval_thresholds_prototype_4feat.log
logs/eval_topk_prototype_4feat.log
```

---

## 23. Final status

```text
ML baseline complete.
Aggregate feature limitation identified.
Need frontend/gem5 per-port directional feature generation next.
```

This is not a failed result.

It is a valid baseline result:

```text
Aggregate temporal router features are excellent for attack detection.
Precise malicious-router localization requires directional/per-port features.
```
