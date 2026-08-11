# V5 P2 Decoder D0 Numerical Root-Cause Audit

## Status

- Status: **COMPLETE**
- Root-cause classification:
  **OVERLY BRITTLE MIP-GAP REPORTING GATE AT MACHINE-EPSILON SCALE**
- Next stage:
  **D1 prospective numerical-certification policy freeze**

## What is established

The frozen decoder checks solver success/status before checking the reported
MIP gap. Therefore, the observed gap exception proves that HiGHS had returned
success with status zero before the strict reporting gate rejected the result.

- Observed reported gap: `2.4064574342771067e-14`
- Old reporting tolerance: `1.4210854715202004e-14`
- Gap/tolerance ratio: `1.693393875670834`
- Observed gap in float64 epsilons: `108.377208`
- Observed gap as a fraction of `1e-12`: `0.024065`

The old gate equals 64 float64 epsilons. The observed residual is finite,
nonnegative, and below the separately frozen `1e-12` semantic tolerance.

## Validation-only audit

- Frozen validation items: **12528**
- Deterministically audited items: **512**
- Repeated deterministic decodes: **16**
- Route and union-mask reconstruction: **PASS**
- Unique attacker-source constraint: **PASS**
- High-precision semantic-face certificate: **PASS**
- Repeated-output determinism: **PASS**
- Maximum sampled primary gap: `0`
- Maximum sampled lexicographic gap: `0`

The previous full validation report is retained as evidence that all 12,528
validation items completed under the old gate.

## Important limitation

The failed test item's MILP vector was not persisted. D0 therefore does not
claim item-specific proof of primal feasibility, integrality, objective
consistency, or route legality for that failed item.

D1 must define a stronger prospective certificate that combines an explicitly
frozen reporting ceiling with independent feasibility, integrality, objective,
and legal-route checks. The value `1e-12` is recorded only as a D1 candidate;
it is **not frozen by D0**.

## Safety boundary

- Test path formed: **false**
- Test directory checked: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Model inference performed: **false**
- Decoder modified: **false**
- Authorization created: **false**

Artifact SHA-256: `671c6524810a765a0fc54116afd1fe3fd2e1599780310f3b379b6920c8708d81`
