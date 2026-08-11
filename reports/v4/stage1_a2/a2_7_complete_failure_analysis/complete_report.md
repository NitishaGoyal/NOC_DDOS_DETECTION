# V4-A2 Complete Failure Analysis and A1 Comparison

Generated: `2026-07-23T13:12:25.222766+00:00`

## Integrity result

- A1 parameters: `882`
- A2 parameters: `2274`
- Threshold and decoder fitting: validation only
- Test role: development-comparison transfer only
- Dataset modification: none

## Capacity verdict

**Outcome A — width improves ranking**

Residual original failure modes remain: `True`

## Key matched comparison

| Metric | A1 | A2 | Δ A2−A1 | Winner |
|---|---:|---:|---:|---|
| graph_f1_at_fpr10 | 0.757091 | 0.857137 | 0.100046 | A2 |
| test_graph_fpr | 0.087627 | 0.088693 | 0.001066 | A1 |
| attack_node_f1 | 0.454197 | 0.682947 | 0.228750 | A2 |
| attack_exact_localization | 0.114603 | 0.400024 | 0.285422 | A2 |
| attacker_count_accuracy | 0.194469 | 0.452710 | 0.258241 | A2 |
| oracle_count_attack_exact | 0.313590 | 0.596668 | 0.283077 | A2 |
| learned_decoder_attack_exact | 0.141128 | 0.400024 | 0.258897 | A2 |
| true_attacker_mean_rank | 4.428263 | 3.346106 | -1.082157 | A2 |
| attack_empty_prediction_fraction | 0.383399 | 0.257435 | -0.125964 | A2 |
| within_two_hops_fp_fraction | 0.891361 | 0.766950 | -0.124411 | A2 |
| corner_node_f1 | 0.297989 | 0.495662 | 0.197673 | A2 |
| parameter_count | 882.000000 | 2274.000000 | 1392.000000 | A1 |

## A3 gate

A3 remains necessary: `True`

Mandatory components supported by the evidence:

- `source_preserving_local_skip`: `True`
- `one_gcn_instead_of_two`: `True`
- `explicit_count_head`: `True`
- `count_conditioned_topk`: `True`
- `hard_negative_neighbour_ranking`: `True`
- `valid_port_masks`: `True`

## Hardware contract

The next model remains a reusable 4×4 regional expert with shared weights across 8×8/16×16 regions, a configurable regional embedding dimension, 16 attacker scores, a graph score, and 5 attacker-count logits.

## Stop condition

A3 is on hold until this report and `a1_vs_a2_capacity_verdict.json` are reviewed. The generated `A3_START_GATE.json` intentionally remains `HOLD_FOR_HUMAN_REVIEW`.
