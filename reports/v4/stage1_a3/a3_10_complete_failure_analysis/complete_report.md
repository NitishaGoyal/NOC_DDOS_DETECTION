# V4-A3 Complete Failure Analysis and A2 Comparison

**Decision:** accepted as complementary localization model

## Key A2 versus A3 metrics

| Metric | A2 | A3 | Delta | Winner |
|---|---:|---:|---:|---|
| graph_f1_at_fpr10 | 0.857137 | 0.893226 | 0.036089 | A3 |
| test_graph_fpr | 0.088693 | 0.113594 | 0.024901 | A2 |
| graph_recall | 0.789902 | 0.862059 | 0.072157 | A3 |
| attack_node_f1 | 0.682947 | 0.710553 | 0.027605 | A3 |
| attack_exact_localization | 0.400024 | 0.473797 | 0.073773 | A3 |
| attacker_count_accuracy | 0.452710 | 0.553283 | 0.100573 | A3 |
| count_mae | 0.818670 | 0.662290 | -0.156380 | A3 |
| oracle_count_attack_exact | 0.596668 | 0.691021 | 0.094354 | A3 |
| learned_decoder_attack_exact | 0.400024 | 0.473797 | 0.073773 | A3 |
| true_attacker_mean_rank | 3.346106 | 2.976858 | -0.369249 | A3 |
| true_attacker_median_rank | 2.000000 | 2.000000 | 0.000000 | tie |
| attack_empty_prediction_fraction | 0.257435 | 0.121445 | -0.135989 | A3 |
| one_hop_fp_fraction | 0.512779 | 0.359780 | -0.152999 | A3 |
| within_two_hops_fp_fraction | 0.766950 | 0.683751 | -0.083199 | A3 |
| corner_node_f1 | 0.495662 | 0.532615 | 0.036953 | A3 |
| edge_node_f1 | 0.728987 | 0.748141 | 0.019154 | A3 |
| interior_node_f1 | 0.713555 | 0.729966 | 0.016411 | A3 |
| corner_interior_f1_gap | 0.217893 | 0.197351 | -0.020542 | A3 |
| validation_to_test_graph_f1_gap_abs | 0.015699 | 0.012270 | -0.003429 | A3 |
| validation_to_test_attack_node_f1_gap_abs | 0.006231 | 0.008364 | 0.002133 | A2 |
| parameter_count | 2274.000000 | 2983.000000 | 709.000000 | A2 |

## Component evidence

- `ranking_loss_reduced_nearby_fp`: `True`
- `valid_masks_reduced_boundary_bias`: `True`
- `one_gcn_local_skip_improved_oracle_ranking`: `True`
- `explicit_count_head_reduced_cardinality_error`: `True`
- `graph_localization_tradeoff_resolved`: `True`

## Integrity

- `validation_only_selection`: `True`
- `test_not_used_for_selection`: `True`
- `checkpoint_hash_matches_thresholds`: `True`
- `parameter_count_verified`: `True`
- `designation_verified`: `True`
- `test_not_independent_holdout`: `True`
- `prediction_alignment`: `True`
- `generic_comprehensive_present`: `True`
- `generic_remaining_present`: `True`
- `a2_comparison_generated`: `True`
- `acceptance_decision_generated`: `True`

The V4 matched-control shortcut remains a publication blocker shared by A1/A2/A3.
