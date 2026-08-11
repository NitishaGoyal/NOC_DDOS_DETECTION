# V4-A4a-1C Full-Training Contract

## Model

- Primary model: A4a-Slot-Full
- Frozen A3 temporal/local/graph encoder and graph head
- Trainable slot decoder parameters: 1,076
- Total deployed parameters: 3,189

## Runtime

- seed: 7
- maximum_epochs: 75
- early_stopping_patience_epochs: 10
- minimum_delta: 0.0001
- optimizer: AdamW
- learning_rate: 0.001
- weight_decay: 0.0001
- gradient_clip_norm: 5.0
- batch_size_primary: 512
- batch_size_fallback: 256
- fallback_requires_authorized_resource_probe_failure: True
- num_workers: 2
- pin_memory: True
- persistent_workers: True
- prefetch_factor: 2
- automatic_mixed_precision: False
- full_validation_every_epochs: 3
- training_split_only_for_optimizer_steps: True
- validation_split_only_for_checkpointing_and_selection: True
- development_test_loader_constructed: False

## Losses

- permutation_invariant_matching_weight: 1.0
- derived_union_membership_bce_weight: 0.5
- derived_cardinality_cross_entropy_weight: 0.5
- duplicate_slot_penalty_weight: 0.1
- null_target_weight: 0.514585764294049
- null_target_weight_source: A4a-1B training-split prevalence only
- null_target_weight_report_sha256: 95e2e34321c827937d5d363192f36e38f06f74a9376e35af7332fd928c031e54
- matching_implementation: vectorized exact enumeration, cross-checked against the audited reference implementation

## Checkpoint and validation

- training_early_stop_monitor: minimum full-validation total slot loss
- save_every_full_validation_checkpoint: True
- primary_model_selection_occurs_after_training: True
- primary_model_selection_source: validation only
- selection_hard_gates: {'attacker_set_precision_min': 0.98, 'candidate_coverage_min': 0.9, 'persistent_normal_run_false_isolation_fraction': 0.0, 'parameter_budget_pass': True, 'latency_budget_pass': True}
- selection_after_hard_gates: maximize stable exact localization, then attack exact localization, then minimize normal window false isolation
- minimum_meaningful_validation_exact_gain_percentage_points: 2.0
- frozen_development_test_transfers_allowed: 1
- post_transfer_retuning_allowed: False
- run_level_paired_bootstrap_required: True

## Ablation order

- A4a-Slot: matching loss only
- A4a-Slot-Full: matching + derived membership + derived count + duplicate penalty
- A4a-TopK control: retrained membership/count decoder with count-controlled top-k
- A4a-Slot-Full+DirectionalProxy only if validation shows the non-proxy slot model is useful

## Prohibitions

- No encoder or graph-head fine-tuning in A4a decoder-only training.
- No development-test loader construction during training.
- No threshold search during optimizer training.
- No test-specific checkpoint, loss-weight or persistence selection.
- No smoke checkpoint may be promoted as the final model.
- No naive duplicate removal as the reference decoder.
- No directional proxy in the primary model.
- No autonomous-isolation claim unless persistent-normal safety gates pass.
