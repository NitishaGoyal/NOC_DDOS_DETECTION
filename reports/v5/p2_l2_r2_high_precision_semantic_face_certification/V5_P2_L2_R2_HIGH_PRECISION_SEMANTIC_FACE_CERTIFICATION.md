# V5 P2 L2-R2 High-Precision Semantic-Face Certification

## Why L2-R2 is required

The L4-R2 log recorded a float64 semantic-face difference of
`2.4549251520511461e-12` against a total float64 certification envelope
of `2e-12`. This is the second distinct
optimum-face boundary failure, so no further validation-derived allowance is
permitted.

## Frozen global rule

- Semantic tie tolerance: **1e-12, unchanged**
- Decimal precision: **80 digits**
- Objective coefficients: **exact binary64 values via Decimal.from_float**
- Candidate acceptance: **high-precision difference <= 1e-12**
- Candidate outside the face: **exact no-good cut, then re-solve**
- Maximum rejected candidates: **1024**
- New numerical tolerance: **none**
- MILP objective, route space, score weights and tie order: **unchanged**

## Diagnostic item 7010

- High-precision certified: **true**
- High-precision difference: `0E-51`
- Rejected candidates: **0**
- Selected routes: `[44, 89, 224]`

## Cache policy

The L4-R2 failure log last printed **7010**, while the
authoritative contiguous completion bitmap contains **7014** provisional
entries. The additional **4** entries are preserved
only as historical evidence and are not silently grandfathered. L4-R3 must recertify
all 12,528 items from index 0 using a separate resumable bitmap. It compares
recomputed R3 outputs with the historical cache and records every changed
output.

## Security state

- P2 test directory enumerated: **false**
- P2 test tensors deserialized: **false**
- Ready for one-shot test: **false**
