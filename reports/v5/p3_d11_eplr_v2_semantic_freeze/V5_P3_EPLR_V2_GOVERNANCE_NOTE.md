# V5-P3 EPLR-V2 governance

## Classification

**V5-P3-1500-D70 Tranche-A Preliminary Diagnostic — EPLR-V2 Preserve-or-Passthrough Design Freeze**

EPLR-V2 is a new design informed by the completed D9/D10 Tranche-A validation
analysis. It is not an evaluated Tranche-A candidate.

## Frozen decision

EPLR-V2 removes endpoint repair entirely.

When Raw source/victim masks admit exactly K legal routes, EPLR-V2 may replace
only transit/path with route-derived union masks. Source and victim remain Raw.

When Raw endpoints are not legally explainable, EPLR-V2 passes through every
Raw role mask unchanged and reports:

`RAW_ENDPOINTS_UNRESOLVED_PASSTHROUGH`

It does not claim a route certificate for that row.

## Calibration

The D8 A-train-only calibration is reused exactly:

- transit weight: 1.0
- path weight: 1.0
- same per-K low-support thresholds
- no recalibration

## Validation boundary

No second EPLR replay on Tranche-A validation is authorized.

Future confirmation must use an independent boundary:
- frozen Tranche-B validation; or
- future fresh A+B validation under the final protocol.

A-test remains sealed.

## Parallel work

H1 quantization, 8x8 wrapper engineering and Tranche-B work remain unaffected.
