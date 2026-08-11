# V5 P2 L2 Exact Decoder Implementation

## Frozen backend

- Backend: **scipy.optimize.milp_with_embedded_HiGHS**
- Python: **3.12.3**
- NumPy: **2.5.1**
- SciPy: **1.18.0**
- Embedded HiGHS: **1.12.0**
- MIP relative gap: **0.0**
- Time limit: **none**
- Node limit: **none**
- Candidate pruning: **false**
- Beam approximation: **false**
- Semantic tie tolerance: **1e-12**

## Exactness policy

The decoder uses one binary MILP covering K1-K4 and all 240 frozen routes.
It first maximizes the structured score, then solves a second MILP restricted
to the semantic optimum face. The second objective applies the frozen tie-break:
lower K, followed by the lexicographically lower sorted route-ID tuple.

Any nonoptimal status, nonzero MIP gap, missing solution, invalid numeric input,
or failure to certify that the second solution remains on the optimum face is a
hard error.

## Verification

- Unit tests run: **10**
- Result: **PASS**
- Restricted MILP versus exhaustive-oracle comparison: **PASS**
- Repeated-run determinism check: **PASS**
- Module SHA-256: `0d6ae843a0503cf6aa347078dda50e570f69cbbffdaa7ee1e4b2dbfd3b7978bf`
- Route-table SHA-256: `19624d95a83114fba1e27e647aab59a0231c97d45053b598692ae62578b321b0`
- Test source SHA-256: `4a4d6310bae07f62d2ed738a95f33afcba80b2c19480e68e2d76585b813f93d8`
- Test log SHA-256: `8a3fd8ef0d524ed41a318b3d4e2f79cb986d1c6e821b5e3309f9a4a1a2cbb4ff`

## Current state

- Structured decoder protocol locked: **true**
- Route library frozen: **true**
- Exact decoder implemented: **true**
- Exact backend frozen: **true**
- Beam decoder implemented: **false**
- Validation predictions exported: **false**
- Raw thresholds frozen: **false**
- Decoder frozen: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Ready for one-shot test: **false**
- Next stage: **V5 P2 L3 hardware beam decoder implementation**
