# V5 P2 D1 Prospective Numerical-Certification Policy

## Status

- Status: **FROZEN**
- Policy ID: `V5P2-D1-6ce8cc7c76b8d2cff367631077d65f20`
- Next stage: **D2 certified decoder implementation**

## Frozen numerical values

| Check | Frozen value |
|---|---:|
| Reported MIP-gap ceiling | `9.9999999999999998e-13` |
| Integrality absolute tolerance | `1.0000000000000001e-09` |
| Variable-bound absolute tolerance | `1.0000000000000001e-09` |
| Linear-constraint absolute tolerance | `1.0000000000000001e-09` |
| Objective absolute tolerance | `1e-10` |
| Objective relative tolerance | `9.9999999999999998e-13` |
| Semantic-face tolerance | `9.9999999999999998e-13` |
| Decimal precision | `80` digits |

The MIP-gap predicate is:

```text
isfinite(gap) and 0.0 <= gap <= 1e-12
```

The observed P2 residual is `2.4064574342771067e-14`, which is
`0.024065` of the new ceiling.

## Critical rule

Acceptance under the new gap ceiling is **not sufficient**. A result is
certified only when every gate passes:

1. solver success and optimal status;
2. finite solution/objective/gap payload;
3. reported gap within `[0, 1e-12]`;
4. independent integrality and variable-bound checks;
5. independent frozen-linear-constraint feasibility;
6. independent objective recomputation;
7. exact legal-route, unique-source, count, and mask reconstruction;
8. frozen semantic-face and lexicographic certification.

## Persistence rule

Neural outputs, Raw metrics, and A0 metrics must be committed immutably before
A1 begins. An A1 technical failure must not erase or invalidate completed Raw
or A0 results.

## Scientific status

For a future P2 run, this policy is explicitly **post hoc and test-informed**.
Any result must be called a *post hoc numerical-recovery evaluation*.

For V6, the policy is prospective only when the implementation, validation
proof, staged evaluator, checkpoint, and thresholds are frozen before V6 test
access.

## Safety boundary

- Test path formed: **false**
- Test directory checked: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Validation logits loaded: **false**
- Inference performed: **false**
- Decoder modified: **false**
- Authorization created: **false**

Contract SHA-256: `b63efbfe1272fb7f6ab922645182fba86e28d7c8f42a70e0f3ac453b88bec71f`
