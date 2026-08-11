# V5 P2 L2-R0 Solver-Gap Numerical Certification

## Reason for correction

At validation item 4,580, HiGHS returned optimal status while SciPy reported a
relative MIP gap of `1.2458674689278237e-16`. This is below one
float64 machine epsilon and is a reporting-roundoff residual, not a material
optimization gap.

## Frozen correction

- Requested HiGHS relative gap remains **0.0**.
- Solver success and status 0 remain mandatory.
- Reported gap must be finite, nonnegative, and no larger than
  **1.4210854715202004e-14** (`64 × float64 epsilon`).
- The semantic tie tolerance remains **9.9999999999999998e-13**.
- Objective, constraints, routes, score, tie-break, and optimum-face score
  verification are unchanged.
- Any material, negative, or nonfinite gap remains a hard error.

## Regression and cache

- Original plus R1 tests: **16 PASS**
- Triggering item rerun: **PASS**
- Primary reported gap: **0**
- Lexicographic reported gap: **0**
- Preserved L4 exact prefix: **4589 / 12,528**
- Cached outputs recomputed: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
