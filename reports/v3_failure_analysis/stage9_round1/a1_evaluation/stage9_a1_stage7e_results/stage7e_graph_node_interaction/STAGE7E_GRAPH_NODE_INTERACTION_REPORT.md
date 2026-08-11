# Stage 7E — Graph-Head versus Node-Head Interaction

Generated: `2026-07-20T10:13:23.233356+00:00`
Script version: `1.0.0`

## Verdict: **PASS — DOMINANT FAILURE MECHANISM IDENTIFIED**

The analysis used saved validation/test prediction exports only. No model inference, training, dataset modification, or gem5 execution was performed.

## Operating points

| model_name | reference_graph_threshold | validation_selected_graph_threshold | node_threshold | node_threshold_source |
| --- | --- | --- | --- | --- |
| conv1d_gcn | 0.5 | 0.6 | 0.5 | saved_reference_0.50; no test tuning |
| tcn_attention_gcn | 0.5 | 0.26 | 0.5 | saved_reference_0.50; no test tuning |
| tcn_maxpool_gcn | 0.5 | 0.41 | 0.5 | saved_reference_0.50; no test tuning |
| tcn_meanpool_gcn | 0.5 | 0.56 | 0.5 | saved_reference_0.50; no test tuning |

## Hard-attack attacker-rank summary

| model_name | run_id | window_count | router12_mean_rank | router12_median_rank | router12_top1_rate | router12_top3_rate | router12_positive_margin_rate | router12_probability_mean | router12_probability_median | router12_probability_minimum | router12_probability_maximum | router12_probability_p05 | router12_probability_p25 | router12_probability_p75 | router12_probability_p95 | router12_mean_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv1d_gcn | N-5-10-Pbursty-R51-A-12-S20-V3 | 3293 | 1.2629820832068024 | 1.0 | 0.8578803522623747 | 0.9635590646826602 | 0.8578803522623747 | 0.6017590252781551 | 0.6749728918075562 | 5.0302394811296836e-05 | 0.9981539845466614 | 0.05307274833321572 | 0.3549600839614868 | 0.873589813709259 | 0.9697183728218078 | 0.4609542380101888 |
| tcn_attention_gcn | N-5-10-Pbursty-R51-A-12-S20-V3 | 3293 | 1.3495293045854844 | 1.0 | 0.8165806255693896 | 0.9456422714849682 | 0.8165806255693896 | 0.33092656155455286 | 0.250617653131485 | 2.24410512394968e-09 | 0.9894310235977173 | 0.028057695552706725 | 0.10332483798265457 | 0.526540219783783 | 0.8392509102821349 | 0.20136630606579528 |
| tcn_maxpool_gcn | N-5-10-Pbursty-R51-A-12-S20-V3 | 3293 | 1.1979957485575463 | 1.0 | 0.9058609170968721 | 0.9568782265411478 | 0.9058609170968721 | 0.3860099297282355 | 0.2890588939189911 | 0.0007861905614845455 | 0.997020423412323 | 0.012447574734687805 | 0.08198745548725128 | 0.6872733235359192 | 0.9601274490356445 | 0.3174294010114491 |
| tcn_meanpool_gcn | N-5-10-Pbursty-R51-A-12-S20-V3 | 3293 | 1.3814151229881566 | 1.0 | 0.8153659277254783 | 0.9268144549043426 | 0.8153659277254783 | 0.3713939369417546 | 0.26379328966140747 | 1.4972511053201742e-05 | 0.9968900084495544 | 0.0008106618188321592 | 0.021965067833662033 | 0.7191295027732849 | 0.9664657235145567 | 0.30544074825674883 |

## Cross-model diagnosis

| model_name | selected_graph_threshold | hard_attack_graph_recall | hard_attack_false_negative_count | router12_top1_rate_all_windows | router12_rank1_given_graph_false_negative | router12_top3_given_graph_false_negative | node_informative_given_graph_false_negative | node_exact_given_graph_false_negative | node_empty_given_graph_false_negative | node_wrong_source_given_graph_false_negative | node_overpredicting_given_graph_false_negative | one_hop_spread_given_graph_false_negative | median_router12_probability_given_graph_false_negative | diagnostic_flags | primary_diagnosis | cross_model_graph_recall_vs_node_top1_correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv1d_gcn | 0.6 | 0.6638323716975403 | 1107 | 0.8578803522623747 | 0.8473351400180669 | 0.953026196928636 | 0.9701897018970189 | 0.4281842818428184 | 0.5076784101174345 | 0.018066847335140017 | 0.04607046070460705 | 0.0 | 0.4600406885147095 | A_graph_readout_failure | A_graph_readout_failure | -0.26412287983117794 |
| tcn_attention_gcn | 0.26 | 0.7643486182812026 | 776 | 0.8165806255693896 | 0.8157216494845361 | 0.9432989690721649 | 0.9432989690721649 | 0.041237113402061855 | 0.9445876288659794 | 0.010309278350515464 | 0.003865979381443299 | 0.0 | 0.08817774429917336 | A_graph_readout_failure | A_graph_readout_failure | -0.26412287983117794 |
| tcn_maxpool_gcn | 0.41 | 0.319465532948679 | 2241 | 0.9058609170968721 | 0.9344042838018741 | 0.9861668897813476 | 0.9968763944667559 | 0.2101740294511379 | 0.7670682730923695 | 0.0071396697902722 | 0.015618027666220438 | 0.000892458723784025 | 0.170789897441864 | A_graph_readout_failure | A_graph_readout_failure | -0.26412287983117794 |
| tcn_meanpool_gcn | 0.56 | 0.2411175220163984 | 2499 | 0.8153659277254783 | 0.8179271708683473 | 0.9339735894357744 | 0.936374549819928 | 0.2909163665466186 | 0.6702681072428972 | 0.035214085634253704 | 0.003601440576230492 | 0.0 | 0.1561138778924942 | A_graph_readout_failure | A_graph_readout_failure | -0.26412287983117794 |

## Temporal blocks

| model_name | operating_point | graph_threshold | temporal_block | window_count | end_epoch_minimum | end_epoch_maximum | graph_recall | mean_graph_probability | mean_router12_probability | router12_top1_rate | router12_top3_rate | graph_wrong_node_informative_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv1d_gcn | reference_0_50 | 0.5 | early | 1098 | 7 | 1104 | 0.7158469945355191 | 0.6482090719132951 | 0.5451447919314129 | 0.7185792349726776 | 0.8989071038251366 | 0.9166666666666666 |
| conv1d_gcn | reference_0_50 | 0.5 | middle | 1098 | 1105 | 2202 | 0.7176684881602914 | 0.6621771017712913 | 0.5943229145676624 | 0.9262295081967213 | 0.9954462659380692 | 0.9903225806451613 |
| conv1d_gcn | reference_0_50 | 0.5 | late | 1097 | 2203 | 3299 | 0.7921604375569735 | 0.6994714917582795 | 0.6658677561576848 | 0.9288969917958068 | 0.9963536918869644 | 0.9956140350877193 |
| conv1d_gcn | validation_selected | 0.6 | early | 1098 | 7 | 1104 | 0.6338797814207651 | 0.6482090719132951 | 0.5451447919314129 | 0.7185792349726776 | 0.8989071038251366 | 0.9303482587064676 |
| conv1d_gcn | validation_selected | 0.6 | middle | 1098 | 1105 | 2202 | 0.644808743169399 | 0.6621771017712913 | 0.5943229145676624 | 0.9262295081967213 | 0.9954462659380692 | 0.9897435897435898 |
| conv1d_gcn | validation_selected | 0.6 | late | 1097 | 2203 | 3299 | 0.7128532360984503 | 0.6994714917582795 | 0.6658677561576848 | 0.9288969917958068 | 0.9963536918869644 | 0.9968253968253968 |
| tcn_attention_gcn | reference_0_50 | 0.5 | early | 1098 | 7 | 1104 | 0.4936247723132969 | 0.5000534630922491 | 0.3779595957378111 | 0.7112932604735883 | 0.8615664845173042 | 0.9514388489208633 |
| tcn_attention_gcn | reference_0_50 | 0.5 | middle | 1098 | 1105 | 2202 | 0.5336976320582878 | 0.5275101480585797 | 0.28355637957046476 | 0.8561020036429873 | 0.97632058287796 | 0.94921875 |
| tcn_attention_gcn | reference_0_50 | 0.5 | late | 1097 | 2203 | 3299 | 0.7274384685505926 | 0.681953636725215 | 0.33126401669157324 | 0.8824065633546034 | 0.9990884229717412 | 1.0 |
| tcn_attention_gcn | validation_selected | 0.26 | early | 1098 | 7 | 1104 | 0.7131147540983607 | 0.5000534630922491 | 0.3779595957378111 | 0.7112932604735883 | 0.8615664845173042 | 0.9428571428571428 |
| tcn_attention_gcn | validation_selected | 0.26 | middle | 1098 | 1105 | 2202 | 0.7112932604735883 | 0.5275101480585797 | 0.28355637957046476 | 0.8561020036429873 | 0.97632058287796 | 0.917981072555205 |
| tcn_attention_gcn | validation_selected | 0.26 | late | 1097 | 2203 | 3299 | 0.8687329079307201 | 0.681953636725215 | 0.33126401669157324 | 0.8824065633546034 | 0.9990884229717412 | 1.0 |
| tcn_maxpool_gcn | reference_0_50 | 0.5 | early | 1098 | 7 | 1104 | 0.2768670309653916 | 0.3239378852747255 | 0.3867838635777958 | 0.7932604735883424 | 0.8715846994535519 | 0.9899244332493703 |
| tcn_maxpool_gcn | reference_0_50 | 0.5 | middle | 1098 | 1105 | 2202 | 0.1830601092896175 | 0.23214890485213452 | 0.34692541677248195 | 0.9426229508196722 | 0.9990892531876139 | 1.0 |
| tcn_maxpool_gcn | reference_0_50 | 0.5 | late | 1097 | 2203 | 3299 | 0.2999088422971741 | 0.3533640466998966 | 0.42435543187828134 | 0.9817684594348223 | 1.0 | 1.0 |
| tcn_maxpool_gcn | validation_selected | 0.41 | early | 1098 | 7 | 1104 | 0.36429872495446264 | 0.3239378852747255 | 0.3867838635777958 | 0.7932604735883424 | 0.8715846994535519 | 0.9899713467048711 |
| tcn_maxpool_gcn | validation_selected | 0.41 | middle | 1098 | 1105 | 2202 | 0.23132969034608378 | 0.23214890485213452 | 0.34692541677248195 | 0.9426229508196722 | 0.9990892531876139 | 1.0 |
| tcn_maxpool_gcn | validation_selected | 0.41 | late | 1097 | 2203 | 3299 | 0.3628076572470374 | 0.3533640466998966 | 0.42435543187828134 | 0.9817684594348223 | 1.0 | 1.0 |
| tcn_meanpool_gcn | reference_0_50 | 0.5 | early | 1098 | 7 | 1104 | 0.2677595628415301 | 0.3181289673078297 | 0.3206698823167754 | 0.6739526411657559 | 0.8242258652094717 | 0.8818407960199005 |
| tcn_meanpool_gcn | reference_0_50 | 0.5 | middle | 1098 | 1105 | 2202 | 0.2122040072859745 | 0.27060123819675264 | 0.3477737717153825 | 0.8779599271402551 | 0.97632058287796 | 0.9722543352601156 |
| tcn_meanpool_gcn | reference_0_50 | 0.5 | late | 1097 | 2203 | 3299 | 0.4038286235186873 | 0.4272884733893432 | 0.4458059272761062 | 0.894257064721969 | 0.9799453053783045 | 0.9755351681957186 |
| tcn_meanpool_gcn | validation_selected | 0.56 | early | 1098 | 7 | 1104 | 0.21402550091074682 | 0.3181289673078297 | 0.3206698823167754 | 0.6739526411657559 | 0.8242258652094717 | 0.8644264194669756 |
| tcn_meanpool_gcn | validation_selected | 0.56 | middle | 1098 | 1105 | 2202 | 0.17304189435336975 | 0.27060123819675264 | 0.3477737717153825 | 0.8779599271402551 | 0.97632058287796 | 0.9724669603524229 |
| tcn_meanpool_gcn | validation_selected | 0.56 | late | 1097 | 2203 | 3299 | 0.33637192342752964 | 0.4272884733893432 | 0.4458059272761062 | 0.894257064721969 | 0.9799453053783045 | 0.9766483516483516 |

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
| same_attacker12_s20_mixed_test | N-1-6-9-14-Pmixed-R52-A-12-S20-V3 | True |  |

## Interpretation rules

- **A — Graph-readout failure:** graph false negatives retain strong rank-1/top-3 attacker evidence.
- **B — Node calibration:** the attacker ranks highly but remains just below the fixed node threshold.
- **C — Spatial smearing:** attacker evidence spreads strongly into one-hop neighbours or extra routers.
- **D — Shared encoder/training:** graph false negatives also have weak or absent true-attacker evidence.
- **E — Multitask inconsistency:** graph and node quality move in opposing directions across models.
- **F — Mixed evidence:** no single mechanism meets the declared diagnostic rule.

These are diagnostic classifications, not causal proofs. Stage 9 interventions should be selected from the observed pattern and then tested in controlled retraining experiments.
