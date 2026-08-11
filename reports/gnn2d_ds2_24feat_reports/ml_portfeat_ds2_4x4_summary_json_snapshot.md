# DS2 4×4 Port-Feature Summary JSON Snapshot

This file records the exact `summary.json` test outputs from the two trained port-feature models.

These values are important because they are the direct saved model summaries, not manually recomputed numbers.

---

## 1. Prototype split summary

Model summary file:

```text
models/temporal_gcn_float_prototype_ds2_4x4_portfeat/summary.json
```

Best epoch:

```text
37
```

Split:

```text
prototype
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

Interpretation:

```text
The prototype split detects attacks well, but graph recall is lower than the placement split.
Node localization is already much stronger than the aggregate-feature baseline.
At the default/fixed threshold used in the training summary, node F1 is 0.8792 and exact localization is 0.5871.
```

---

## 2. Placement split summary

Model summary file:

```text
models/temporal_gcn_float_placement_ds2_4x4_portfeat/summary.json
```

Best epoch:

```text
58
```

Split:

```text
placement
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

Interpretation:

```text
The placement split result is the strongest clean result.
It shows excellent graph-level generalization and strong node-level localization on unseen attacker placements.
At the default/fixed threshold used in the training summary, graph F1 is 0.9990, node F1 is 0.9572, and exact localization is 0.7798.
```

---

## 3. Why this snapshot matters

This snapshot should be kept because it records the exact saved output from:

```text
models/temporal_gcn_float_prototype_ds2_4x4_portfeat/summary.json
models/temporal_gcn_float_placement_ds2_4x4_portfeat/summary.json
```

These are the model-level summary artifacts that should be cited whenever reporting the final port-feature baseline.

The most important values are:

```text
Prototype:
- best epoch: 37
- graph F1: 0.9525
- node F1: 0.8792
- exact localization: 0.5871

Placement:
- best epoch: 58
- graph F1: 0.9990
- node F1: 0.9572
- exact localization: 0.7798
```

The placement result should be treated as the stronger paper-facing result because it evaluates generalization across attacker placement.
