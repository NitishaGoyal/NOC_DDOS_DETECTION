# V4-A4a-0S Frozen Contract

## Primary decoder

Four unordered attacker slots are used. Each slot predicts one of 17 classes: router 0–15 or NULL.

- Slot count: 4
- Classes per slot: 17
- Slot latent dimension: 8
- New trainable parameters: 1076
- Total deployed parameters: 3189

The primary predicted set comes from a unique constrained assignment. NULL may be reused; a real router may be assigned to at most one slot.

## Training

- Encoder frozen: True
- Graph head frozen: True
- Maximum epochs: 75
- Early-stopping patience: 10
- Seed: 7

## Validation selection

- hard_gate_persistent_normal_run_false_isolation_fraction: 0.0
- hard_gate_attacker_set_precision_min: 0.98
- hard_gate_candidate_coverage_min: 0.9
- minimum_exact_improvement_over_a3_h32_percentage_points: 2.0
- selection_after_hard_gates: maximize stable exact localization, then attack exact localization, then minimize normal window false isolation
- a3_h32_reference_exact: 0.7936002680515999
- a3_h32_oracle_count_ceiling: 0.905511811023622

## Kill criteria

- Reject further A4a development if the best validation-selected slot model improves attack exact localization by less than 2 percentage points over A3.11-H32.
- Reject any A4a model that increases persistent normal-run false isolation.
- Reject a gain achieved only through material candidate-coverage collapse below 0.90.
- Reject the structured slot design if it does not outperform the retrained A4a-TopK ablation.
- After a kill decision, close V4 architecture tuning and move to V5 rather than introducing development-test-specific fixes.

## Frozen-transfer promotion criteria

- desirable_exact_improvement_percentage_points_min: 3.0
- strong_exact_improvement_percentage_points: 3 to 5
- attacker_set_precision_min: 0.98
- persistent_normal_run_false_isolation_fraction: 0.0
- candidate_coverage_must_not_collapse: True
- parameter_budget_must_pass: True
- latency_budget_must_pass: True
- no_post_transfer_retuning: True

## Claim boundary

The strongest defensible V4 claim is high-precision candidate localization with temporally stable attacker-set estimation. Fully autonomous isolation remains conditional on strict persistent-normal safety gates.
