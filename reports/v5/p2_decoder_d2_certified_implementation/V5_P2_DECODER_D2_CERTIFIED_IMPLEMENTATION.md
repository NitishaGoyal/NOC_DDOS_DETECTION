# V5 P2 Decoder D2 Certified Implementation

## Status

- Status: **COMPLETE**
- D1 policy: `V5P2-D1-6ce8cc7c76b8d2cff367631077d65f20`
- Legacy decoder modified: **false**
- Next stage: **D3 full validation certification proof**

## Implementation

The versioned certified decoder adds independent solver-payload, gap,
integrality, bound, linear-feasibility, objective, and legal-route
certificates while preserving the frozen structured score and tie-break
semantics.

- Certified source SHA-256: `30309376cb0990a0f482045e2dac413d4370729cf94f9e556f895af2c5c8c71d`
- Certified test SHA-256: `d7f67d784ab71c910e09277b5fe608b2541afe09ef67b8f4776300db33ddfaf4`

## Validation-only smoke equivalence

- Items audited: **32 / 12528**
- Legacy/certified semantic equivalence: **PASS**
- All primary and lexicographic results certified: **PASS**
- Repeated deterministic decodes: **4 PASS**
- Maximum primary reported gap: `0`
- Maximum lexicographic reported gap: `0`
- Maximum integrality error: `4.9562522941772896e-13`
- Maximum bound violation: `4.75689570057247e-13`
- Maximum linear-constraint violation: `4.3471330818670135e-13`
- Maximum objective difference: `3.5527136788005009e-15`

## Safety boundary

- Test path formed: **false**
- Test directory checked/enumerated: **false**
- Test tensors deserialized: **false**
- Model/checkpoint loaded: **false**
- Neural inference performed: **false**
- Evaluation authorization created: **false**

Artifact SHA-256: `fcb78ae4059f49e4c78d4e3cf603160f7d8bad72857d6ef6b777cc4de0ab7055`
