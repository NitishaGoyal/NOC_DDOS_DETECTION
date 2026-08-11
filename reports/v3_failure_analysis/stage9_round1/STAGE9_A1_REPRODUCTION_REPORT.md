# Stage 9 A1 Conv1D Baseline Reproduction

- Generated: `2026-07-18T19:45:00.891070+00:00`
- Script version: `1.0.0`
- Training exit code: `0`
- Provisional verdict: **PASS WITH POSSIBLE NONDETERMINISM — RUN-LEVEL DIAGNOSTICS REQUIRED**

## Purpose

A1 reran the original Conv1D-TemporalGCN baseline while keeping the dataset, V3 split, architecture, optimizer, class weights, losses, checkpoint score, thresholds, and seed unchanged.

## Preflight

- Frozen artifacts matched: `True`
- Dataset shapes matched: `True`
- Original V3 split matched generated split exactly: `True`
- Parameter count: `882`
- Graph positive weight: `0.45161290322580644`
- Node positive weight: `13.4`
- CUDA device: `NVIDIA GeForce RTX 4070 Laptop GPU`

## Training result

- Best epoch: `92`
- Best validation score: `1.8273760228644256`
- Epochs completed: `100`
- Early stopping: `False`
- Stopped epoch: `None`

## Metric comparison

| Split | Metric | Original | A1 | Delta | Tolerance | Pass |
|---|---:|---:|---:|---:|---:|---:|
| val | f1 | 0.951087700 | 0.957308709 | +0.006221009 | ±0.005 | False |
| test | f1 | 0.838374737 | 0.862864804 | +0.024490066 | ±0.010 | False |
| val | node_f1 | 0.875693322 | 0.870067314 | -0.005626008 | ±0.010 | True |
| test | node_f1 | 0.726200991 | 0.719416091 | -0.006784899 | ±0.010 | True |
| val | exact_localization | 0.737531827 | 0.725501647 | -0.012030181 | ±0.015 | True |
| test | exact_localization | 0.531009834 | 0.515522437 | -0.015487398 | ±0.020 | True |

## Hard configuration checks

| Check | Pass |
|---|---:|
| model_name_matches | True |
| parameter_count_882 | True |
| split_sizes_match | True |
| graph_pos_weight_matches | True |
| node_pos_weight_matches | True |
| graph_threshold_0_5 | True |
| node_threshold_0_5 | True |

## Interpretation

The pipeline configuration matches, but at least one summary metric exceeded its tolerance. Run-level diagnostics are required to determine whether this is ordinary nondeterminism or a deeper reproducibility issue.

## Next step

Freeze the A1 outputs, export aligned graph and node probabilities, evaluate the reference and validation-selected graph thresholds, and repeat the run-level/Stage 7E diagnostics.
