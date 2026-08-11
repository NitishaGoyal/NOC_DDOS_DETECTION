# Stage 7E — Graph-Head versus Node-Head Interaction

Generated: `2026-07-18T18:52:49.190219+00:00`
Script version: `1.0.0`

## Verdict: **PASS — DOMINANT FAILURE MECHANISM IDENTIFIED**

The analysis used saved validation/test prediction exports only. No model inference, training, dataset modification, or gem5 execution was performed.

## Operating points

| model_name | reference_graph_threshold | validation_selected_graph_threshold | node_threshold | node_threshold_source |
| --- | --- | --- | --- | --- |
| conv1d_gcn | 0.5 | 0.47 | 0.5 | saved_reference_0.50; no test tuning |
| tcn_attention_gcn | 0.5 | 0.48 | 0.5 | saved_reference_0.50; no test tuning |
| tcn_maxpool_gcn | 0.5 | 0.46 | 0.5 | saved_reference_0.50; no test tuning |
| tcn_meanpool_gcn | 0.5 | 0.49 | 0.5 | saved_reference_0.50; no test tuning |

## Hard-attack attacker-rank summary

| model_name | run_id | window_count | router12_mean_rank | router12_median_rank | router12_top1_rate | router12_top3_rate | router12_positive_margin_rate | router12_probability_mean | router12_probability_median | router12_probability_minimum | router12_probability_maximum | router12_probability_p05 | router12_probability_p25 | router12_probability_p75 | router12_probability_p95 | router12_mean_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv1d_gcn | N-5-10-Pbursty-R51-A-12-S20-V3 | 9 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.6765920122464498 | 0.6870341897010803 | 0.6145556569099426 | 0.7817057371139526 | 0.6196006417274476 | 0.6320652961730957 | 0.7052194476127625 | 0.7531743764877319 | 0.5603117520610491 |
| tcn_attention_gcn | N-5-10-Pbursty-R51-A-12-S20-V3 | 9 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.6086186303032769 | 0.5938268303871155 | 0.5676934719085693 | 0.7288159728050232 | 0.5718803644180298 | 0.5918403267860413 | 0.5995475053787231 | 0.6891316294670105 | 0.46716394358211094 |
| tcn_maxpool_gcn | N-5-10-Pbursty-R51-A-12-S20-V3 | 9 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.511474672291014 | 0.5022549033164978 | 0.4455369710922241 | 0.5688763856887817 | 0.4542857050895691 | 0.48771363496780396 | 0.5364518761634827 | 0.564984917640686 | 0.37982548193799126 |
| tcn_meanpool_gcn | N-5-10-Pbursty-R51-A-12-S20-V3 | 9 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.45126376549402875 | 0.4648270010948181 | 0.3594339191913605 | 0.532703697681427 | 0.37732771039009094 | 0.42144328355789185 | 0.4785127341747284 | 0.5134109497070313 | 0.3144140425655577 |

## Cross-model diagnosis

| model_name | selected_graph_threshold | hard_attack_graph_recall | hard_attack_false_negative_count | router12_top1_rate_all_windows | router12_rank1_given_graph_false_negative | router12_top3_given_graph_false_negative | node_informative_given_graph_false_negative | node_exact_given_graph_false_negative | node_empty_given_graph_false_negative | node_wrong_source_given_graph_false_negative | node_overpredicting_given_graph_false_negative | one_hop_spread_given_graph_false_negative | median_router12_probability_given_graph_false_negative | diagnostic_flags | primary_diagnosis | cross_model_graph_recall_vs_node_top1_correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv1d_gcn | 0.47 | 0.0 | 9 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.6870341897010803 | A_graph_readout_failure | A_graph_readout_failure |  |
| tcn_attention_gcn | 0.48 | 0.3333333333333333 | 6 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5936644375324249 | A_graph_readout_failure | A_graph_readout_failure |  |
| tcn_maxpool_gcn | 0.46 | 0.7777777777777778 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5640120506286621 | A_graph_readout_failure | A_graph_readout_failure |  |
| tcn_meanpool_gcn | 0.49 | 0.7777777777777778 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.45295755565166473 | A_graph_readout_failure;B_node_threshold_calibration | A_graph_readout_failure |  |

## Temporal blocks

| model_name | operating_point | graph_threshold | temporal_block | window_count | end_epoch_minimum | end_epoch_maximum | graph_recall | mean_graph_probability | mean_router12_probability | router12_top1_rate | router12_top3_rate | graph_wrong_node_informative_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv1d_gcn | reference_0_50 | 0.5 | early | 3 | 7 | 9 | 0.0 | 0.34052229921023053 | 0.6686568458875021 | 1.0 | 1.0 | 1.0 |
| conv1d_gcn | reference_0_50 | 0.5 | middle | 3 | 10 | 12 | 0.0 | 0.2902718385060628 | 0.708049456278483 | 1.0 | 1.0 | 1.0 |
| conv1d_gcn | reference_0_50 | 0.5 | late | 3 | 13 | 15 | 0.0 | 0.33889053265253705 | 0.6530697345733643 | 1.0 | 1.0 | 1.0 |
| conv1d_gcn | validation_selected | 0.47 | early | 3 | 7 | 9 | 0.0 | 0.34052229921023053 | 0.6686568458875021 | 1.0 | 1.0 | 1.0 |
| conv1d_gcn | validation_selected | 0.47 | middle | 3 | 10 | 12 | 0.0 | 0.2902718385060628 | 0.708049456278483 | 1.0 | 1.0 | 1.0 |
| conv1d_gcn | validation_selected | 0.47 | late | 3 | 13 | 15 | 0.0 | 0.33889053265253705 | 0.6530697345733643 | 1.0 | 1.0 | 1.0 |
| tcn_attention_gcn | reference_0_50 | 0.5 | early | 3 | 7 | 9 | 0.0 | 0.44800181190172833 | 0.5972914298375448 | 1.0 | 1.0 | 1.0 |
| tcn_attention_gcn | reference_0_50 | 0.5 | middle | 3 | 10 | 12 | 0.0 | 0.4335843622684479 | 0.633601168791453 | 1.0 | 1.0 | 1.0 |
| tcn_attention_gcn | reference_0_50 | 0.5 | late | 3 | 13 | 15 | 0.3333333333333333 | 0.4850465754667918 | 0.5949632922808329 | 1.0 | 1.0 | 1.0 |
| tcn_attention_gcn | validation_selected | 0.48 | early | 3 | 7 | 9 | 0.3333333333333333 | 0.44800181190172833 | 0.5972914298375448 | 1.0 | 1.0 | 1.0 |
| tcn_attention_gcn | validation_selected | 0.48 | middle | 3 | 10 | 12 | 0.3333333333333333 | 0.4335843622684479 | 0.633601168791453 | 1.0 | 1.0 | 1.0 |
| tcn_attention_gcn | validation_selected | 0.48 | late | 3 | 13 | 15 | 0.3333333333333333 | 0.4850465754667918 | 0.5949632922808329 | 1.0 | 1.0 | 1.0 |
| tcn_maxpool_gcn | reference_0_50 | 0.5 | early | 3 | 7 | 9 | 0.0 | 0.4849940339724223 | 0.5015793740749359 | 1.0 | 1.0 | 1.0 |
| tcn_maxpool_gcn | reference_0_50 | 0.5 | middle | 3 | 10 | 12 | 0.6666666666666666 | 0.4791923562685649 | 0.5310139656066895 | 1.0 | 1.0 | 1.0 |
| tcn_maxpool_gcn | reference_0_50 | 0.5 | late | 3 | 13 | 15 | 0.0 | 0.46013996998469037 | 0.5018306771914164 | 1.0 | 1.0 | 1.0 |
| tcn_maxpool_gcn | validation_selected | 0.46 | early | 3 | 7 | 9 | 1.0 | 0.4849940339724223 | 0.5015793740749359 | 1.0 | 1.0 |  |
| tcn_maxpool_gcn | validation_selected | 0.46 | middle | 3 | 10 | 12 | 0.6666666666666666 | 0.4791923562685649 | 0.5310139656066895 | 1.0 | 1.0 | 1.0 |
| tcn_maxpool_gcn | validation_selected | 0.46 | late | 3 | 13 | 15 | 0.6666666666666666 | 0.46013996998469037 | 0.5018306771914164 | 1.0 | 1.0 | 1.0 |
| tcn_meanpool_gcn | reference_0_50 | 0.5 | early | 3 | 7 | 9 | 0.6666666666666666 | 0.5318829715251923 | 0.49765756726264954 | 1.0 | 1.0 | 1.0 |
| tcn_meanpool_gcn | reference_0_50 | 0.5 | middle | 3 | 10 | 12 | 0.6666666666666666 | 0.5938221116860708 | 0.4069643517335256 | 1.0 | 1.0 | 1.0 |
| tcn_meanpool_gcn | reference_0_50 | 0.5 | late | 3 | 13 | 15 | 1.0 | 0.6490302483240763 | 0.44916937748591107 | 1.0 | 1.0 |  |
| tcn_meanpool_gcn | validation_selected | 0.49 | early | 3 | 7 | 9 | 0.6666666666666666 | 0.5318829715251923 | 0.49765756726264954 | 1.0 | 1.0 | 1.0 |
| tcn_meanpool_gcn | validation_selected | 0.49 | middle | 3 | 10 | 12 | 0.6666666666666666 | 0.5938221116860708 | 0.4069643517335256 | 1.0 | 1.0 | 1.0 |
| tcn_meanpool_gcn | validation_selected | 0.49 | late | 3 | 13 | 15 | 1.0 | 0.6490302483240763 | 0.44916937748591107 | 1.0 | 1.0 |  |

## Control availability

| control_role | run_id | present_in_export | missing_reason |
| --- | --- | --- | --- |
| hard_attack | N-5-10-Pbursty-R51-A-12-S20-V3 | True |  |
| hard_normal | N-3-7-8-12-Pmixed-R18-V3 | True |  |
| exact_benign_background | N-5-10-Pbursty-R16-V3 | True |  |
| same_attacker12_s20_bursty_train_candidate | N-0-6-Pbursty-R24-A-12-S20-V3 | False | run is absent from exported val/test cohort |
| same_attacker12_s20_stream_train_candidate | N-0-15-Pstream-R20-A-12-S20-V3 | False | run is absent from exported val/test cohort |
| same_attacker12_s20_stream_test | N-2-13-Pstream-R50-A-12-S20-V3 | True |  |
| same_attacker12_s20_mixed_train_candidate | N-0-6-9-15-Pmixed-R26-A-12-S20-V3 | False | run is absent from exported val/test cohort |
| same_attacker12_s20_mixed_test | N-1-6-9-14-Pmixed-R52-A-12-S20-V3 | False | run is absent from exported val/test cohort |

## Interpretation rules

- **A — Graph-readout failure:** graph false negatives retain strong rank-1/top-3 attacker evidence.
- **B — Node calibration:** the attacker ranks highly but remains just below the fixed node threshold.
- **C — Spatial smearing:** attacker evidence spreads strongly into one-hop neighbours or extra routers.
- **D — Shared encoder/training:** graph false negatives also have weak or absent true-attacker evidence.
- **E — Multitask inconsistency:** graph and node quality move in opposing directions across models.
- **F — Mixed evidence:** no single mechanism meets the declared diagnostic rule.

These are diagnostic classifications, not causal proofs. Stage 9 interventions should be selected from the observed pattern and then tested in controlled retraining experiments.
