# V5 P2 Decoder D3 Full Validation Certification Proof

## Status

- Status: **COMPLETE**
- Policy: `V5P2-D1-6ce8cc7c76b8d2cff367631077d65f20`
- Certified validation items: **12528/12528**
- Primary MILPs certified: **12528/12528**
- Lexicographic MILPs certified: **12528/12528**
- Deterministic replay: **64/64 PASS**

## Numerical maxima

| Certificate quantity | Maximum |
|---|---:|
| Primary reported MIP gap | `1.6734456776828493e-15` |
| Lexicographic reported MIP gap | `0` |
| Integrality error | `5.201378625579624e-12` |
| Variable-bound violation | `3.7867106886033407e-12` |
| Linear-constraint violation | `6.3096194935496897e-12` |
| Objective difference | `7.1054273576010019e-15` |

- Positive reported-gap certificates:
  **2**
- Certificates above the old 64-epsilon gate:
  **0**
- High-precision candidate rejections:
  **0**
- Validation hypotheses classified as attack after the already frozen A1
  threshold: **5811/12528**

## Interpretation

D3 proves that every frozen validation-logit item can be decoded under the D1
policy with independent primary and lexicographic numerical certificates.

This is a decoder-correctness and numerical-stability result. It is not a model
performance result and is not a test result. No threshold was selected or
changed.

## Immutable output archive

- Chunks: **126**
- Rows: **12528**
- Chunk-hash sequence SHA-256:
  `a9b5d78960adff96a0eb1ab8e34b22ca5a77ba943d6ff451c5ab6f2c747ff285`
- Semantic-fingerprint sequence SHA-256:
  `bb8c1ba0a67ddb6eaedf116138f3d7ac6c4af5f77ee3d21ce982c781235cc4a0`

## Safety boundary

- Test path formed: **false**
- Test directory checked/enumerated: **false**
- Test tensors deserialized: **false**
- Model/checkpoint loaded: **false**
- Neural inference performed: **false**
- Threshold selection performed: **false**
- Evaluation authorization created: **false**

## Next stage

**V5 P2 evaluator D4 staged Raw/A0/A1 implementation and validation**

Artifact SHA-256: `11339c6afcec5275a310219e05a360638121cdfabd2f02589273e2c84af56def`
