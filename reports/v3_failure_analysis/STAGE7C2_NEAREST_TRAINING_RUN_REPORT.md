# Stage 7C2 — Nearest Training-Run and Class-Margin Analysis

Stage 7C2 used the frozen Stage 7C1 V2 all-training scaling reference. 
No validation/test value influenced scaling, feature selection, metric choice, 
representation weighting, neighbour count or class-margin definition.

R6 raw-distribution coverage was excluded from the primary geometry because 
its exact-boundary histogram interpretation remains unresolved.

## Fixed geometry

- D1: root mean squared difference divided by the all-training standard deviation.
- D2: root mean squared difference divided by the all-training IQR.
- D3: mean absolute difference divided by the all-training standard deviation.
- Constant training dimensions contribute zero when matched and a fixed finite 
penalty of 5 when violated.
- R1–R5 are measured separately.
- The combined distance is the equal-weight mean of the five representation distances.
- Class margin = nearest opposite-class distance minus nearest same-class distance.
- Positive margin means same-class training is nearer.
- Negative margin means opposite-class training is nearer.

## Dominant-run verdict

- Hard normal: **same class is closer under every predeclared combined distance**.
- Hard attack: **same class is closer under every predeclared combined distance**.

## Combined margins

| Target | Metric | Nearest same | Same distance | Nearest opposite | Opposite distance | Margin |
|---|---|---|---:|---|---:|---:|
| `N-3-7-8-12-Pmixed-R18-V3` | D1_standardized_rmse | `N-0-6-9-15-Pmixed-R6-V3` | 0.3173 | `N-3-12-Pstream-R22-A-1-7-S63-V3` | 0.3549 | +0.0376 |
| `N-3-7-8-12-Pmixed-R18-V3` | D2_robust_iqr_rmse | `N-0-6-9-15-Pmixed-R6-V3` | 0.4711 | `N-3-12-Pstream-R22-A-1-7-S63-V3` | 0.5129 | +0.0418 |
| `N-3-7-8-12-Pmixed-R18-V3` | D3_standardized_manhattan | `N-0-6-9-15-Pmixed-R6-V3` | 0.1818 | `N-3-12-Pstream-R22-A-1-7-S63-V3` | 0.2045 | +0.0227 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | D1_standardized_rmse | `N-0-6-Pbursty-R24-A-11-S20-V3` | 0.2257 | `N-2-5-10-13-Pmixed-R7-V3` | 0.6442 | +0.4184 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | D2_robust_iqr_rmse | `N-3-12-Pstream-R22-A-11-S20-V3` | 0.4619 | `N-2-5-10-13-Pmixed-R7-V3` | 0.8532 | +0.3913 |
| `N-5-10-Pbursty-R51-A-12-S20-V3` | D3_standardized_manhattan | `N-0-6-Pbursty-R24-A-11-S20-V3` | 0.1329 | `N-2-5-10-13-Pmixed-R7-V3` | 0.4749 | +0.3420 |

## Block stability

Each dominant run was divided into eight deterministic contiguous blocks. 
Block summaries used the frozen full-training scaling and full-run training vectors; 
no block-specific reference was fitted.

- `N-3-7-8-12-Pmixed-R18-V3`: 3/8 blocks same-class closer; 5/8 blocks opposite-class closer.
- `N-5-10-Pbursty-R51-A-12-S20-V3`: 8/8 blocks same-class closer; 0/8 blocks opposite-class closer.

## Missing-scenario evidence

- `N-3-7-8-12-Pmixed-R18-V3`: same_class_closer. Nearest same-class `N-0-6-9-15-Pmixed-R6-V3`; nearest opposite-class `N-3-12-Pstream-R22-A-1-7-S63-V3`. Add benign configurations around the target placement and its nearest attack-like neighbourhood; preserve mixed traffic and router-local count/IFD structure.
- `N-5-10-Pbursty-R51-A-12-S20-V3`: same_class_closer. Nearest same-class `N-0-6-Pbursty-R24-A-11-S20-V3`; nearest opposite-class `N-2-5-10-13-Pmixed-R7-V3`. Add weak attacks matching the target profile, active-core placement, attacker placement and count–IFD structure, especially where normal training neighbours are closer.

## Interpretation boundary

- Nearest-run geometry describes the Stage 7C1 summary space; it is not a causal model.
- Direction names are not interpreted physically before builder recovery.
- R6 is excluded from the primary conclusion.
- No classifier was fitted and no model inference was rerun.
- V4 scenarios are suggested as evidence, not finalized here.

Plots created: **12**.
