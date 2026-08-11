# Stage 7C1 V2 — Training Distribution Coverage

All fitted references were constructed using the training split only. 
Validation and test targets were then compared against frozen full-training, same-class and opposite-class references.

Nearest-run ranking, classifier fitting, inference and model training were not performed.

## Training reference

- Training samples represented: **148185**.
- Training runs represented: **45**.
- Selected validation/test targets evaluated: **10**.
- Plots created: **12**.

## Interpretation rules

- Min–max inclusion is not treated as sufficient evidence of coverage.
- p01–p99, p05–p95, z-score, robust-IQR deviation and constant-dimension violations are reported separately.
- Direction names are retained only as fixed feature labels until builder recovery confirms physical mapping.
- Exact-1.0 pattern coverage can establish recurrence or novelty, not the encoding mechanism.

## Dominant-run coverage summary

| Target | Representation | Reference | Outside p01–p99 | Outside min–max | Median |z| | Median robust deviation |
|---|---|---|---:|---:|---:|---:|
| `N-3-7-8-12-Pmixed-R18-V3` | R1_global_run_feature | all_training | 0.0000 | 0.0000 | 0.364 | 0.247 |
| `N-3-7-8-12-Pmixed-R18-V3` | R1_global_run_feature | opposite_class | 0.0000 | 0.0000 | 0.482 | 0.284 |
| `N-3-7-8-12-Pmixed-R18-V3` | R1_global_run_feature | same_class | 0.0152 | 0.0152 | 0.733 | 0.862 |
| `N-3-7-8-12-Pmixed-R18-V3` | R2_feature_family | all_training | 0.0000 | 0.0000 | 0.331 | 0.149 |
| `N-3-7-8-12-Pmixed-R18-V3` | R2_feature_family | opposite_class | 0.0000 | 0.0000 | 0.364 | 0.129 |
| `N-3-7-8-12-Pmixed-R18-V3` | R2_feature_family | same_class | 0.0000 | 0.0000 | 0.752 | 0.897 |
| `N-3-7-8-12-Pmixed-R18-V3` | R3_router_feature | all_training | 0.0281 | 0.0182 | 0.389 | 0.279 |
| `N-3-7-8-12-Pmixed-R18-V3` | R3_router_feature | opposite_class | 0.0495 | 0.0469 | 0.635 | 0.400 |
| `N-3-7-8-12-Pmixed-R18-V3` | R3_router_feature | same_class | 0.0495 | 0.0354 | 0.634 | 0.592 |
| `N-3-7-8-12-Pmixed-R18-V3` | R4_temporal_position_feature | all_training | 0.0000 | 0.0000 | 0.393 | 0.224 |
| `N-3-7-8-12-Pmixed-R18-V3` | R4_temporal_position_feature | opposite_class | 0.0000 | 0.0000 | 0.425 | 0.253 |
| `N-3-7-8-12-Pmixed-R18-V3` | R4_temporal_position_feature | same_class | 0.0333 | 0.0333 | 0.741 | 0.861 |
| `N-3-7-8-12-Pmixed-R18-V3` | R5_spatial_concentration | all_training | 0.0208 | 0.0000 | 0.360 | 0.251 |
| `N-3-7-8-12-Pmixed-R18-V3` | R5_spatial_concentration | opposite_class | 0.0417 | 0.0417 | 0.511 | 0.309 |
| `N-3-7-8-12-Pmixed-R18-V3` | R5_spatial_concentration | same_class | 0.0417 | 0.0417 | 0.625 | 0.799 |
| `N-3-7-8-12-Pmixed-R18-V3` | R6_raw_global_feature | all_training | 0.2556 | 0.0000 | 0.868 | 0.351 |
| `N-3-7-8-12-Pmixed-R18-V3` | R6_raw_global_feature | opposite_class | 0.2562 | 0.0000 | 0.802 | 0.414 |
| `N-3-7-8-12-Pmixed-R18-V3` | R6_raw_global_feature | same_class | 0.2550 | 0.0000 | 0.721 | 0.459 |
| `N-3-7-8-12-Pmixed-R18-V3` | R6_raw_router_feature | all_training | 0.2564 | 0.0000 | 0.620 | 0.365 |
| `N-3-7-8-12-Pmixed-R18-V3` | R6_raw_router_feature | opposite_class | 0.2602 | 0.0000 | 0.621 | 0.449 |
| `N-3-7-8-12-Pmixed-R18-V3` | R6_raw_router_feature | same_class | 0.2554 | 0.0000 | 0.675 | 0.353 |
| `N-3-7-8-12-Pmixed-R18-V3` | R6_raw_temporal_feature | all_training | 0.2556 | 0.0000 | 0.868 | 0.351 |
| `N-3-7-8-12-Pmixed-R18-V3` | R6_raw_temporal_feature | opposite_class | 0.2562 | 0.0000 | 0.802 | 0.414 |
| `N-3-7-8-12-Pmixed-R18-V3` | R6_raw_temporal_feature | same_class | 0.2550 | 0.0000 | 0.721 | 0.457 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R1_global_run_feature | all_training | 0.0000 | 0.0000 | 0.336 | 0.493 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R1_global_run_feature | opposite_class | 0.0076 | 0.0076 | 0.301 | 0.348 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R1_global_run_feature | same_class | 0.1212 | 0.1061 | 1.123 | 0.645 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R2_feature_family | all_training | 0.0000 | 0.0000 | 0.455 | 0.529 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R2_feature_family | opposite_class | 0.0000 | 0.0000 | 0.260 | 0.335 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R2_feature_family | same_class | 0.2576 | 0.2424 | 1.116 | 0.571 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R3_router_feature | all_training | 0.0089 | 0.0052 | 0.465 | 0.441 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R3_router_feature | opposite_class | 0.0224 | 0.0172 | 0.379 | 0.377 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R3_router_feature | same_class | 0.0979 | 0.0943 | 1.014 | 0.646 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R4_temporal_position_feature | all_training | 0.0000 | 0.0000 | 0.318 | 0.463 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R4_temporal_position_feature | opposite_class | 0.0167 | 0.0167 | 0.408 | 0.382 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R4_temporal_position_feature | same_class | 0.0958 | 0.0958 | 1.045 | 0.658 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R5_spatial_concentration | all_training | 0.0000 | 0.0000 | 0.313 | 0.487 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R5_spatial_concentration | opposite_class | 0.0208 | 0.0208 | 0.252 | 0.444 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R5_spatial_concentration | same_class | 0.0729 | 0.0521 | 1.012 | 0.650 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R6_raw_global_feature | all_training | 0.2806 | 0.0000 | 0.756 | 0.367 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R6_raw_global_feature | opposite_class | 0.2805 | 0.0000 | 0.710 | 0.271 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R6_raw_global_feature | same_class | 0.2845 | 0.0000 | 0.766 | 0.426 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R6_raw_router_feature | all_training | 0.2807 | 0.0000 | 0.608 | 0.405 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R6_raw_router_feature | opposite_class | 0.2808 | 0.0000 | 0.632 | 0.301 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R6_raw_router_feature | same_class | 0.2932 | 0.0000 | 0.662 | 0.501 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R6_raw_temporal_feature | all_training | 0.2806 | 0.0000 | 0.756 | 0.367 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R6_raw_temporal_feature | opposite_class | 0.2805 | 0.0000 | 0.710 | 0.271 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | R6_raw_temporal_feature | same_class | 0.2845 | 0.0000 | 0.766 | 0.426 |

## Most novel same-class dimensions

### `N-3-7-8-12-Pmixed-R18-V3`
#### R1_global_run_feature
- `in_count_norm_local` / `mean_spatial_variance` / global: target=0.00048, training median=0.00024, outside p01–p99=False, |robust deviation|=1.926.
- `in_count_norm_local` / `std` / global: target=0.02220, training median=0.01582, outside p01–p99=False, |robust deviation|=1.680.
- `in_count_norm_local` / `p95` / global: target=0.06800, training median=0.04800, outside p01–p99=False, |robust deviation|=1.569.
- `ifd_in_norm_local` / `p05` / global: target=0.01386, training median=0.01838, outside p01–p99=False, |robust deviation|=1.538.
- `in_count_norm_north` / `mean_spatial_variance` / global: target=0.00031, training median=0.00013, outside p01–p99=False, |robust deviation|=1.295.
- `out_count_norm_south` / `mean_spatial_variance` / global: target=0.00031, training median=0.00013, outside p01–p99=False, |robust deviation|=1.295.
- `ifd_in_norm` / `p05` / global: target=0.02094, training median=0.02664, outside p01–p99=False, |robust deviation|=1.286.
- `out_count_norm_local` / `mean_spatial_variance` / global: target=0.00042, training median=0.00021, outside p01–p99=False, |robust deviation|=1.278.
#### R3_router_feature
- `ifd_in_norm_local` / `mean_temporal_excursion` / router 8: target=0.03933, training median=0.87411, outside p01–p99=True, |robust deviation|=10.107.
- `ifd_out_norm_local` / `mean_temporal_excursion` / router 8: target=0.03340, training median=0.89771, outside p01–p99=True, |robust deviation|=6.918.
- `ifd_out_norm_local` / `mean_temporal_excursion` / router 7: target=0.03362, training median=0.89669, outside p01–p99=False, |robust deviation|=6.022.
- `in_count_norm_local` / `mean` / router 8: target=0.03775, training median=0.00529, outside p01–p99=True, |robust deviation|=5.853.
- `in_count_norm_local` / `mean` / router 7: target=0.03779, training median=0.00534, outside p01–p99=False, |robust deviation|=5.807.
- `out_count_norm_local` / `mean` / router 8: target=0.04454, training median=0.00736, outside p01–p99=True, |robust deviation|=5.382.
- `out_count_norm_local` / `mean` / router 7: target=0.04430, training median=0.00742, outside p01–p99=False, |robust deviation|=5.330.
- `out_count_norm_east` / `mean` / router 8: target=0.02589, training median=0.00291, outside p01–p99=True, |robust deviation|=4.844.
#### R4_temporal_position_feature
- `in_count_norm_local` / `mean_spatial_variance` / position 1: target=0.00048, training median=0.00024, outside p01–p99=False, |robust deviation|=1.927.
- `in_count_norm_local` / `mean_spatial_variance` / position 4: target=0.00048, training median=0.00024, outside p01–p99=False, |robust deviation|=1.927.
- `in_count_norm_local` / `mean_spatial_variance` / position 3: target=0.00048, training median=0.00024, outside p01–p99=False, |robust deviation|=1.926.
- `in_count_norm_local` / `mean_spatial_variance` / position 2: target=0.00048, training median=0.00024, outside p01–p99=False, |robust deviation|=1.926.
- `in_count_norm_local` / `mean_spatial_variance` / position 6: target=0.00048, training median=0.00024, outside p01–p99=False, |robust deviation|=1.926.
- `in_count_norm_local` / `mean_spatial_variance` / position 0: target=0.00048, training median=0.00024, outside p01–p99=False, |robust deviation|=1.926.
- `in_count_norm_local` / `mean_spatial_variance` / position 5: target=0.00048, training median=0.00024, outside p01–p99=False, |robust deviation|=1.926.
- `in_count_norm_local` / `mean_spatial_variance` / position 7: target=0.00048, training median=0.00024, outside p01–p99=False, |robust deviation|=1.926.

### `N-5-10-Pbursty-R51-A-12-S20-V3`
#### R1_global_run_feature
- `ifd_in_norm_east` / `p05` / global: target=0.02743, training median=0.02104, outside p01–p99=False, |robust deviation|=1.302.
- `ifd_out_norm_west` / `p05` / global: target=0.02741, training median=0.02102, outside p01–p99=False, |robust deviation|=1.293.
- `in_count_norm_south` / `mean_temporal_variance` / global: target=0.00006, training median=0.00010, outside p01–p99=True, |robust deviation|=1.204.
- `out_count_norm_north` / `mean_temporal_variance` / global: target=0.00006, training median=0.00010, outside p01–p99=True, |robust deviation|=1.204.
- `in_count_norm_east` / `std` / global: target=0.01141, training median=0.01574, outside p01–p99=True, |robust deviation|=1.148.
- `out_count_norm_west` / `std` / global: target=0.01141, training median=0.01574, outside p01–p99=True, |robust deviation|=1.148.
- `ifd_in_norm_west` / `median` / global: target=0.18708, training median=0.10115, outside p01–p99=False, |robust deviation|=1.115.
- `ifd_out_norm_east` / `median` / global: target=0.18689, training median=0.10125, outside p01–p99=False, |robust deviation|=1.114.
#### R3_router_feature
- `ifd_in_norm_west` / `fraction_above_0_95` / router 2: target=0.09535, training median=0.00076, outside p01–p99=True, |robust deviation|=89.000.
- `ifd_out_norm_east` / `fraction_above_0_95` / router 1: target=0.09475, training median=0.00076, outside p01–p99=True, |robust deviation|=88.429.
- `ifd_in_norm_west` / `fraction_equal_1` / router 2: target=0.09414, training median=0.00061, outside p01–p99=True, |robust deviation|=88.000.
- `ifd_out_norm_east` / `fraction_equal_1` / router 1: target=0.09323, training median=0.00061, outside p01–p99=True, |robust deviation|=87.143.
- `ifd_in_norm_west` / `mean_temporal_variance` / router 2: target=0.06107, training median=0.00084, outside p01–p99=True, |robust deviation|=68.135.
- `ifd_out_norm_east` / `mean_temporal_variance` / router 1: target=0.06088, training median=0.00084, outside p01–p99=True, |robust deviation|=67.944.
- `ifd_out_norm` / `fraction_above_0_95` / router 7: target=0.00759, training median=0.00000, outside p01–p99=True, |robust deviation|=22.222.
- `ifd_out_norm` / `fraction_equal_1` / router 7: target=0.00729, training median=0.00000, outside p01–p99=True, |robust deviation|=21.333.
#### R4_temporal_position_feature
- `in_count_norm_south` / `mean_absolute_change_from_previous_position` / position 1: target=0.00777, training median=0.01022, outside p01–p99=True, |robust deviation|=1.041.
- `in_count_norm_south` / `mean_absolute_change_from_previous_position` / position 2: target=0.00777, training median=0.01021, outside p01–p99=True, |robust deviation|=1.040.
- `in_count_norm_south` / `mean_absolute_change_from_previous_position` / position 7: target=0.00777, training median=0.01022, outside p01–p99=True, |robust deviation|=1.040.
- `in_count_norm_south` / `mean_absolute_change_from_previous_position` / position 3: target=0.00777, training median=0.01021, outside p01–p99=True, |robust deviation|=1.040.
- `out_count_norm_north` / `mean_absolute_change_from_previous_position` / position 1: target=0.00777, training median=0.01021, outside p01–p99=True, |robust deviation|=1.039.
- `in_count_norm_south` / `mean_absolute_change_from_previous_position` / position 6: target=0.00777, training median=0.01022, outside p01–p99=True, |robust deviation|=1.039.
- `in_count_norm_south` / `mean_absolute_change_from_previous_position` / position 4: target=0.00777, training median=0.01021, outside p01–p99=True, |robust deviation|=1.039.
- `in_count_norm_south` / `mean_absolute_change_from_previous_position` / position 5: target=0.00777, training median=0.01022, outside p01–p99=True, |robust deviation|=1.038.

## Suspicious router-12 exact-1.0 pattern

- `ifd_in_norm_north`: target exact-1.0 rate=1.000; training median=1.000; training p95=1.000; training runs with the same constant-one pattern=45.
- `ifd_in_norm_west`: target exact-1.0 rate=1.000; training median=1.000; training p95=1.000; training runs with the same constant-one pattern=45.
- `ifd_out_norm_north`: target exact-1.0 rate=1.000; training median=1.000; training p95=1.000; training runs with the same constant-one pattern=45.
- `ifd_out_norm_west`: target exact-1.0 rate=1.000; training median=1.000; training p95=1.000; training runs with the same constant-one pattern=45.

## Hypothesis evidence

| Hypothesis | Stage 7C1 status | Builder recovery required |
|---|---|---:|
| H1 — structured_router_direction_saturation | supported | True |
| H2R — feature_relation_or_spatial_context | inconclusive | False |
| H3 — pooling_loss | not_tested | False |
| H4 — spatial_placement_pattern | supported | True |
| H5 — normalization_or_upper_bound_encoding | inconclusive | True |
| H6 — invalid_or_no_traffic_port_encoding | supported_as_encoding_suspicion | True |
| H7 — count_ifd_relationship | not_tested_univariate | False |
| H8 — router_direction_training_coverage | supported | False |

## Boundary

- Stage 7C1 establishes normalized-dimension coverage or novelty.
- It does not establish nearest training runs or same/opposite-class distance margins.
- It does not test joint count–IFD geometry.
- It does not identify the meaning of exact 1.0 or zero-count values.
- Builder and normalization recovery should proceed in parallel.

## Next stage

Stage 7C2 should use these frozen references to compute standardized nearest-training-run distances and same-class versus opposite-class margins.
