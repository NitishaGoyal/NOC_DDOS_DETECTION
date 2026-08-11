# V5-P3 Final One-Shot Blind-Test Result

**Status:** FINAL / one-shot sealed-test result locked. No test rerun or post-test
retuning is authorized.

## Frozen system

- Production checkpoint: seed 107, best epoch 5
- Base graph threshold: 0.5
- Temporal alarm: P2 (2-of-3 Raw positives)
- Runtime structured decoder: EPLR-V2
- Alarm/localization alignment: L1 same-run one-window causal carry

## Blind-test results

| Metric | Raw neural | Final system |
|---|---:|---:|
| Graph accuracy | 0.80742853 | 0.80981086 |
| Graph F1 | 0.72284275 | 0.72458314 |
| Graph FPR | 0.23642489 | 0.23186683 |
| Graph AUROC | 0.88252466 | — |
| Graph AP | 0.62155743 | — |
| Strict all-task exactness | 0.63900520 | 0.75227404 |

Final true-positive-alarm all-count-and-role exactness:
`0.77001876`

P2 episode detection rate:
`1.00000000`

P2 median first-detection delay:
`0.0 windows`

P2 p95 additional delay versus Raw:
`2.0 windows`

## Frozen reporting deltas

- P2 graph accuracy vs Raw: **+0.238233 percentage points**
- P2 graph F1 vs Raw: **+0.174039 percentage points**
- P2 graph FPR vs Raw: **-0.455807 percentage points**
- Relative FPR reduction: **1.9279%**
- Final strict exactness vs Raw: **+11.326884 percentage points**

## Latency

- Observation window: `32` epochs
- Neural inference: `0.00006763 s/window`
  amortized at frozen batch-256 CUDA evaluation
- EPLR-V2 mean software decode:
  `6.98537680562122e-05 s/window`
- L1 carry horizon: `1` window

These are kept as separate latency components; they are not collapsed into one
physical-time latency without an independent epoch-duration/clock mapping.

## Transparent limitation

The final false-positive P2 alarm nonempty-endpoint payload exposure rate is
`1.0`.

This does not create additional graph false alarms; it means that when the
already-frozen P2 system produces a false confirmed alarm, its actionable
localization payload may also be nonempty.

## Governance

The sealed test has been consumed. From this point, the reported one-shot result
is descriptive/final. Any architecture, threshold, decoder, persistence, or
alignment improvement is future work and requires a new predeclared experiment
rather than re-evaluation on this sealed test.
