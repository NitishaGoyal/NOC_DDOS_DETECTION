# V5 P2 Task-D Formal Selection Evidence

## Validation and security checks

- Completed runs: 10/10
- Candidates: Conv1D and GraphConv
- Seeds: 107, 117, 127, 137, 147
- Common frozen B1 protocol: verified
- Common loader hash: verified
- AMP: disabled for every run
- Threshold tuning: not performed
- Test directory enumerated: false
- Test tensors deserialized: false
- Architecture selected: false

## Aggregate comparison

| Metric | Conv1D mean ± population SD | GraphConv mean ± population SD | GraphConv − Conv1D |
|---|---:|---:|---:|
| Selection score | 0.769377 ± 0.004353 | 0.776654 ± 0.003471 | +0.007277 |
| Graph AUROC | 0.892441 ± 0.003511 | 0.893390 ± 0.003502 | +0.000949 |
| Graph AP | 0.701920 ± 0.014883 | 0.710352 ± 0.016551 | +0.008432 |
| Graph F1 at 0.5 | 0.740660 ± 0.001780 | 0.713983 ± 0.015146 | -0.026677 |
| Graph FPR at 0.5 | 0.257940 ± 0.011695 | 0.204314 ± 0.044210 | -0.053625 |
| Attacker-count macro F1 | 0.998110 ± 0.001832 | 0.994861 ± 0.004741 | -0.003249 |
| Source AP | 0.641648 ± 0.013733 | 0.633823 ± 0.010038 | -0.007825 |
| Transit AP | 0.590587 ± 0.006637 | 0.604135 ± 0.012365 | +0.013548 |
| Victim AP | 0.608388 ± 0.008930 | 0.645249 ± 0.008466 | +0.036861 |
| Path AP | 0.625780 ± 0.005768 | 0.645340 ± 0.007680 | +0.019561 |
| Macro role AP | 0.616601 ± 0.004990 | 0.632137 ± 0.006673 | +0.015536 |
| Validation loss | 1.170962 ± 0.015032 | 1.325561 ± 0.114493 | +0.154598 |
| Latency, microseconds/item | 50.075426 ± 0.120276 | 55.321291 ± 1.965742 | +5.245864 |
| Training elapsed seconds | 1671.657037 ± 285.324175 | 1884.938710 ± 871.112046 | +213.281673 |

## Paired-seed comparison

| Metric | Mean paired delta | GraphConv seed wins | Conv1D seed wins | Ties |
|---|---:|---:|---:|---:|
| Selection score | +0.007277 | 4 | 1 | 0 |
| Graph AUROC | +0.000949 | 4 | 1 | 0 |
| Graph AP | +0.008432 | 4 | 1 | 0 |
| Graph F1 at 0.5 | -0.026677 | 0 | 5 | 0 |
| Graph FPR at 0.5 | -0.053625 | 4 | 1 | 0 |
| Attacker-count macro F1 | -0.003249 | 1 | 4 | 0 |
| Source AP | -0.007825 | 2 | 3 | 0 |
| Transit AP | +0.013548 | 3 | 2 | 0 |
| Victim AP | +0.036861 | 5 | 0 | 0 |
| Path AP | +0.019561 | 5 | 0 | 0 |
| Macro role AP | +0.015536 | 5 | 0 | 0 |
| Validation loss | +0.154598 | 1 | 4 | 0 |
| Latency, microseconds/item | +5.245864 | 0 | 5 | 0 |
| Training elapsed seconds | +213.281673 | 2 | 3 | 0 |

## Efficiency comparison

- Conv1D parameters: 43,273
- GraphConv parameters: 59,785
- Parameter increase: 38.16%
- Conv1D linear MAC proxy: 10,899,776
- GraphConv linear MAC proxy: 11,161,920
- Linear-MAC increase: 2.41%
- GraphConv graph-message scalar operations: 6,144

## Current result

- Primary-metric provisional leader: **graphconv**
- Architecture selected: **false**
- This file records evidence only and does not perform post-hoc automatic architecture selection.
