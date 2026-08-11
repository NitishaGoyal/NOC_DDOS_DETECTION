# V4-A1 Remaining Diagnostics Report

- Analysis pass: **True**
- Model inference performed: **False**
- Test used for fitting/selection: **False**

## Decoder decision

- Selected validation method: **frozen_threshold**
- Fraction of validation oracle gap recovered: **0.000**
- Recommended next model action: **representation_ranking_dominates_build_source_preserving_model**
- Plain A2 decision: **plain_a2_low_priority**
- More GCN layers: **do_not_add_more_gcn_layers; neighbour spreading is already a dominant failure**

## Interpretation gates

- decoder_gap_recovery_ge_0_50: `False`
- decoder_gap_recovery_ge_0_20: `False`
- topology_calibration_gain_ge_0_03: `False`
- oracle_test_attack_exact_below_0_50: `False`
- ranking_representation_problem_remains: `False`

## Hard limitations

- The dataset retains the confirmed matched-control active-core shortcut.
- Saved tensors do not contain explicit valid-port/clipping masks.
- Temporal max-pooling causality cannot be proven without an encoder ablation.
- Mean-readout causality cannot be proven without saved embeddings or a readout ablation.
- The test split is a development-comparison set, not an independent publication holdout.
