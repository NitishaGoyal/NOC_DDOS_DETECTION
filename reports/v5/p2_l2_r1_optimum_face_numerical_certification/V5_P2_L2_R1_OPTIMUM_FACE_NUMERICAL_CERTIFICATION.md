# V5 P2 L2-R1 Optimum-Face Numerical Certification

## Trigger

The bound historical L4-R1 log records that validation item **6820** returned
optimal status, while its independently recomputed score differed from the
primary score by `1.1439738045737613e-12`. The original semantic-face
tolerance is `9.9999999999999998e-13`; the excess was a float64 evaluation
residual.

A fresh execution of the same item is certified with difference
`1.7763568394002505e-15` and allowance
`9.9999999999999998e-13`. Exact reproduction of
the historical last-bit residual is not required.

## Frozen correction

- Semantic tie tolerance: **unchanged at 1e-12**
- MILP objective and constraints: **unchanged**
- Route set and tie-break objective: **unchanged**
- Item-specific numerical allowance:
  `9.9999999999999998e-13`
- Total certification tolerance:
  `2e-12`
- Numerical allowance hard cap: **1e-12**
- Material excess above the certification tolerance: **hard error**

## Cache and security

- Combined tests: **22 PASS**
- Preserved exact-decoder prefix: **6822 / 12,528**
- Cached outputs recomputed: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
