# V4-A1 Remaining Diagnostics Report

- Analysis pass: **True**
- Model inference performed: **False**
- Test used for fitting/selection: **False**

## Decoder decision

- Selected validation method: **topology_calibrated_learned_count_topk**
- Fraction of validation oracle gap recovered: **0.260**
- Recommended next model action: **run_plain_a2_only_as_one_seed_capacity_probe_then_structural_model**
- Plain A2 decision: **optional_secondary_capacity_probe**
- More GCN layers: **do_not_add_more_gcn_layers; neighbour spreading is already a dominant failure**

## Interpretation gates

- decoder_gap_recovery_ge_0_50: `False`
- decoder_gap_recovery_ge_0_20: `True`
- topology_calibration_gain_ge_0_03: `False`
- oracle_test_attack_exact_below_0_50: `True`
- ranking_representation_problem_remains: `True`

## Hard limitations

- The dataset retains the confirmed matched-control active-core shortcut.
- Saved tensors do not contain explicit valid-port/clipping masks.
- Temporal max-pooling causality cannot be proven without an encoder ablation.
- Mean-readout causality cannot be proven without saved embeddings or a readout ablation.
- The test split is a development-comparison set, not an independent publication holdout.
