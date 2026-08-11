# Stage 7B — Feature and Saturation Analysis

Stage 7B scanned the normalized V3 feature tensor one run at a time. It produced compact summaries for all 71 runs and detailed router, temporal, directional and spatial summaries for the selected matched runs.

The analysis reports observed normalized behaviour only. It does not establish the provenance or cause of clipping until the dataset builder and normalization logic are recovered.

## Scan

- Samples scanned: **233803**.
- Runs summarized: **71**.
- Features summarized per run: **24**.
- Detailed selected runs: **10**.
- Plots created: **15**.
- Inference rerun: **False**.
- Training performed: **False**.

## Metadata corrections applied

- Features 4–13 were explicitly classified as directional input/output flit-count features.
- Stage 7A was not rewritten; Stage 7B records the corrected taxonomy as a new provenance table.
- `true_attacker_mask` is available in prediction exports and will be used later in Stage 7E. No node grouping was performed here.

## Preliminary hypothesis status

| Hypothesis | Status | Primary metric | Causal claim allowed |
|---|---|---|---:|
| H1 — directional_ifd_saturation | inconclusive | mean_directional_ifd_fraction_equal_1 | False |
| H2 — weak_temporal_signal | contradicted | mean_temporal_range_across_features | False |
| H3 — pooling_loss_precursor | inconclusive | sparse_single_position_excursion_fraction | False |
| H4 — spatial_placement_pattern | inconclusive | mean_router_std_across_features | False |
| H5 — normalization_upper_bound_symptom | inconclusive | directional_ifd_fraction_equal_1_severity_spread | False |

## Strongest matched-pair feature differences

### hard_normal_minus_exact_normal
- `in_count_norm_east`: standardized mean difference +0.195; exact-1.0-rate difference +0.000; temporal-range difference +0.002.
- `out_count_norm_west`: standardized mean difference +0.195; exact-1.0-rate difference +0.000; temporal-range difference +0.002.
- `ifd_in_norm`: standardized mean difference -0.192; exact-1.0-rate difference -0.002; temporal-range difference -0.016.
- `out_count_norm_east`: standardized mean difference +0.190; exact-1.0-rate difference +0.000; temporal-range difference +0.002.
- `in_count_norm_west`: standardized mean difference +0.190; exact-1.0-rate difference +0.000; temporal-range difference +0.002.

### hard_normal_minus_validation_normal
- `ifd_in_norm`: standardized mean difference -0.231; exact-1.0-rate difference -0.001; temporal-range difference -0.014.
- `in_count_norm_east`: standardized mean difference +0.211; exact-1.0-rate difference +0.000; temporal-range difference +0.002.
- `out_count_norm_west`: standardized mean difference +0.211; exact-1.0-rate difference +0.000; temporal-range difference +0.002.
- `ifd_out_norm`: standardized mean difference -0.211; exact-1.0-rate difference -0.001; temporal-range difference -0.012.
- `out_count_norm_east`: standardized mean difference +0.176; exact-1.0-rate difference +0.000; temporal-range difference +0.002.

### hard_attack_minus_stream_A12_S20
- `ifd_in_norm_west`: standardized mean difference -0.134; exact-1.0-rate difference -0.052; temporal-range difference -0.029.
- `ifd_out_norm_east`: standardized mean difference -0.134; exact-1.0-rate difference -0.052; temporal-range difference -0.029.
- `in_count_norm_south`: standardized mean difference -0.125; exact-1.0-rate difference +0.000; temporal-range difference -0.001.
- `out_count_norm_north`: standardized mean difference -0.125; exact-1.0-rate difference +0.000; temporal-range difference -0.001.
- `in_count_norm_north`: standardized mean difference -0.125; exact-1.0-rate difference +0.000; temporal-range difference -0.001.

### hard_attack_minus_mixed_A12_S20
- `input_flit_count_norm`: standardized mean difference -1.272; exact-1.0-rate difference +0.000; temporal-range difference -0.002.
- `output_flit_count_norm`: standardized mean difference -1.272; exact-1.0-rate difference +0.000; temporal-range difference -0.002.
- `ifd_out_norm`: standardized mean difference +0.712; exact-1.0-rate difference +0.001; temporal-range difference +0.038.
- `ifd_in_norm`: standardized mean difference +0.681; exact-1.0-rate difference +0.002; temporal-range difference +0.040.
- `out_count_norm_local`: standardized mean difference -0.635; exact-1.0-rate difference +0.000; temporal-range difference -0.003.

### S20_minus_S63
- `output_flit_count_norm`: standardized mean difference -0.583; exact-1.0-rate difference +0.000; temporal-range difference -0.001.
- `input_flit_count_norm`: standardized mean difference -0.583; exact-1.0-rate difference +0.000; temporal-range difference -0.001.
- `ifd_out_norm`: standardized mean difference +0.392; exact-1.0-rate difference +0.001; temporal-range difference +0.026.
- `ifd_in_norm`: standardized mean difference +0.383; exact-1.0-rate difference +0.002; temporal-range difference +0.028.
- `ifd_out_norm_local`: standardized mean difference +0.331; exact-1.0-rate difference +0.112; temporal-range difference +0.004.

### S20_minus_S77
- `output_flit_count_norm`: standardized mean difference -1.264; exact-1.0-rate difference +0.000; temporal-range difference -0.001.
- `input_flit_count_norm`: standardized mean difference -1.264; exact-1.0-rate difference +0.000; temporal-range difference -0.001.
- `ifd_out_norm`: standardized mean difference +0.724; exact-1.0-rate difference +0.001; temporal-range difference +0.041.
- `ifd_in_norm`: standardized mean difference +0.702; exact-1.0-rate difference +0.002; temporal-range difference +0.044.
- `out_count_norm_local`: standardized mean difference -0.607; exact-1.0-rate difference +0.000; temporal-range difference -0.002.

## Interpretation boundary

- A high exact-1.0 rate is described as observed normalized saturation.
- Stage 7B cannot determine whether 1.0 came from clipping, a true physical maximum, or a default encoding.
- Sparse temporal excursions may support a weak-signal explanation, but they do not by themselves prove that mean or max pooling caused the model failure.
- Spatial differences support a placement-pattern hypothesis, but route-level causation requires routing and builder recovery.

## Next stage

Stage 7C should use training runs only to define coverage ranges, standardization, nearest-run distances and out-of-training-range rates.
