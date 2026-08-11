# Stage 7A — Dataset and Prediction Alignment

Stage 7A establishes a leakage-free index map between the original V3 feature tensor, graph/node labels, all 71 runs, the train/validation/test split, Stage 3 prediction exports and Stage 6 diagnostic groups.

No feature statistics, classifier fitting, inference or retraining were performed.

## Paths

- Dataset root: `/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3`
- Failure-analysis root: `/home/zira/research/projects/GNN-2d/reports/v3_failure_analysis`

## Tensor

- `x.npy` shape: `(233803, 16, 8, 24)`
- `x.npy` dtype: `float32`
- Memory mapped: `True`
- Feature count: `24`

## Split reconstruction

- Split source: `dataset_split_array`
- Dataset split cross-check passed: `True`

| Split | Samples | Runs | Minimum windows/run | Maximum windows/run |
|---|---:|---:|---:|---:|
| train | 148185 | 45 | 3293 | 3293 |
| val | 42809 | 13 | 3293 | 3293 |
| test | 42809 | 13 | 3293 | 3293 |

## Selected Stage 7 groups

| Group | Role | Run | Split | Class | Profile | Active cores | Attackers | Strength |
|---|---|---|---|---:|---|---|---|---:|
| A_hard_normal_exact_test | target | `N-3-7-8-12-Pmixed-R18-V3` | test | 0 | mixed | 3-7-8-12 |  | 0 |
| A_hard_normal_exact_test | control | `N-1-6-9-14-Pmixed-R17-V3` | test | 0 | mixed | 1-6-9-14 |  | 0 |
| B_hard_normal_validation_analogue | target | `N-3-7-8-12-Pmixed-R18-V3` | test | 0 | mixed | 3-7-8-12 |  | 0 |
| B_hard_normal_validation_analogue | control | `N-2-6-10-14-Pmixed-R13-V3` | val | 0 | mixed | 2-6-10-14 |  | 0 |
| C_hard_attack_exact_A12_S20 | target | `N-5-10-Pbursty-R51-A-12-S20-V3` | test | 1 | bursty | 5-10 | 12 | 20 |
| C_hard_attack_exact_A12_S20 | control_stream | `N-2-13-Pstream-R50-A-12-S20-V3` | test | 1 | stream | 2-13 | 12 | 20 |
| C_hard_attack_exact_A12_S20 | control_mixed | `N-1-6-9-14-Pmixed-R52-A-12-S20-V3` | test | 1 | mixed | 1-6-9-14 | 12 | 20 |
| D_hard_attack_bursty_S20 | target | `N-5-10-Pbursty-R51-A-12-S20-V3` | test | 1 | bursty | 5-10 | 12 | 20 |
| D_hard_attack_bursty_S20 | control | `N-4-15-Pbursty-R41-A-11-S20-V3` | val | 1 | bursty | 4-15 | 11 | 20 |
| E_same_bursty_background_severity | S20_target | `N-5-10-Pbursty-R51-A-12-S20-V3` | test | 1 | bursty | 5-10 | 12 | 20 |
| E_same_bursty_background_severity | S63_control | `N-5-10-Pbursty-R51-A-1-7-S63-V3` | test | 1 | bursty | 5-10 | 1-7 | 63 |
| E_same_bursty_background_severity | S77_control | `N-5-10-Pbursty-R51-A-1-11-12-S77-V3` | test | 1 | bursty | 5-10 | 1-11-12 | 77 |
| F_broad_bursty_background | normal_control | `N-5-10-Pbursty-R16-V3` | test | 0 | bursty | 5-10 |  | 0 |
| F_broad_bursty_background | S20_target | `N-5-10-Pbursty-R51-A-12-S20-V3` | test | 1 | bursty | 5-10 | 12 | 20 |

## Node export status

| Model | Labels | Probabilities | Logits | Predictions | Limited Stage 7E grouping possible |
|---|---:|---:|---:|---:|---:|
| conv1d_gcn | False | True | True | False | True |
| tcn_attention_gcn | False | True | True | False | True |
| tcn_meanpool_gcn | False | True | True | False | True |
| tcn_maxpool_gcn | False | True | True | False | True |

## Topology

- `edge_index.npy` available: `True`
- Row-major 4×4 mesh confirmed: `True`
- Router 12 neighbours: `8-13`

## Stage 6 alignment

- Stage 6 dominant-run agreement aligned: `True`

## Leakage boundary

- Training samples are explicitly identified before any summary fitting.
- Stage 7C training ranges and distance standardization must use only `split=train` rows.
- Stage 7D classifier fitting must use training runs only; limited regularization choices may use validation runs only.
- Test runs remain frozen diagnostic evaluation cases.
- Stage 7A performs no data-derived scaling and selects no features.

## Next step

After this alignment passes, Stage 7B may compute global, router, temporal, directional and saturation summaries from the normalized tensor.
