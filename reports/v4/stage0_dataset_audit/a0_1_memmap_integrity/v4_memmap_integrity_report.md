# V4-A0.1 — Memmap Integrity and Structural Array Audit

- Generated: `2026-07-22T18:44:11.625390+00:00`
- Script version: `1.0.0`
- Dataset: `/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v4_all16_chrono_memmap`
- Verdict: **V4_A0_1_MEMMAP_INTEGRITY_PASS**

## Safety boundary

- Full model inference performed: **False**
- Model training performed: **False**
- Threshold selection performed: **False**
- Dataset tree modified: **False**
- Complete `x.npy` loaded into RAM: **False**
- Validation/test samples structurally inspected: **True**
- Validation/test predictions accessed: **False**

## Hard checks

- [PASS] `all_required_files_present`
- [PASS] `all_arrays_shape_dtype_memmap_valid`
- [PASS] `x_all_values_finite`
- [PASS] `x_all_values_in_documented_range_0_1`
- [PASS] `y_graph_binary`
- [PASS] `y_node_binary`
- [PASS] `all_routers_have_attacker_labels`
- [PASS] `graph_sample_counts_match_expected`
- [PASS] `graph_run_counts_match_expected`
- [PASS] `attacker_count_run_distribution_matches_expected`
- [PASS] `split_code_map_contains_train_val_test`
- [PASS] `split_run_counts_match_expected`
- [PASS] `split_sample_counts_match_expected`
- [PASS] `run_segment_count_matches_expected`
- [PASS] `unique_run_count_matches_expected`
- [PASS] `run_indices_are_expected_sequence`
- [PASS] `no_repeated_run_segments`
- [PASS] `all_runs_have_expected_samples`
- [PASS] `all_end_epoch_sequences_valid`
- [PASS] `run_level_metadata_constant`
- [PASS] `graph_labels_match_attacker_counts`
- [PASS] `node_labels_match_attacker_counts`
- [PASS] `completed_runs_count_matches_expected`
- [PASS] `completed_runs_unique`
- [PASS] `edge_shape_valid`
- [PASS] `edge_ids_valid`
- [PASS] `edge_count_and_uniqueness_valid`
- [PASS] `edge_self_loop_count_valid`
- [PASS] `edge_nonself_count_valid`
- [PASS] `edge_reciprocal`
- [PASS] `edge_connected`
- [PASS] `edge_degrees_match_mesh`
- [PASS] `edge_exact_row_major_mesh_match`
- [PASS] `source_root_metadata_unchanged`

## Documented remaining gates

A pass here authorizes only V4-A0.2 manifest-to-array alignment.
It does not authorize model training.

