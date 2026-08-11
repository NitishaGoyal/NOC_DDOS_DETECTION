# V5 P2 D5 Finalized-Architecture E1 Integration

## Status

- Status: **COMPLETE**
- Final architecture:
  **Causal Depthwise-Separable Conv1D Temporal Encoder + Two-Layer GraphConv**
- Checkpoint: **seed 107, epoch 25**
- Validation items regenerated: **12528/12528**
- Validation batches: **49/49**

## Exact E1 reproduction

- Six neural-logit arrays bitwise identical: **PASS**
- Labels, role mask, item order, and manifest-row order identical: **PASS**
- Physical-port mask identical: **PASS**
- Stable-item sequence verified: **PASS**
- First-batch repeated inference bitwise identical: **PASS**

The actual finalized model, frozen checkpoint, audited PRIMARY58 loader,
physical-port mask, static 4×4 edge index, and pair-block validation order
therefore reproduce the immutable E1 archive exactly.

## Staged-pipeline equivalence

- Raw outputs bitwise identical to D4 E2: **PASS**
- Raw metrics identical to D4 E2: **PASS**
- A0 outputs bitwise identical to D4 E3: **PASS**
- A0 metrics identical to D4 E3: **PASS**
- Certified A1 outputs bitwise identical to D4 E4: **PASS**
- Certified A1 metrics identical to D4 E4: **PASS**

D3 is transitively bound to the actual finalized model because D3 consumed the
immutable E1 logits and D5 reproduced every one of those logits bitwise.

## Validation metrics

| Group | Graph accuracy | Strict all-task exactness |
|---|---:|---:|
| Raw | `0.794141` | `0.612947` |
| A0 | `0.794141` | `0.637692` |
| Certified A1 | `0.802123` | `0.731003` |

These remain validation metrics, not test results.

## Safety boundary

- Validation model inference performed: **true**
- Test path formed: **false**
- Test directory checked/enumerated: **false**
- Test tensor files opened: **false**
- Test tensors deserialized: **false**
- Threshold/model/checkpoint/decoder selection performed: **false**
- Decoder execution performed: **false**
- Evaluation authorization created: **false**

Any future P2 evaluation remains explicitly post hoc and test-informed.

## Next stage

**V5 P2 D6 post-hoc numerical-recovery readiness and disclosure freeze**

Artifact SHA-256: `6fe42ee47b6b126b76203e94bb7c3cf3a655541506c9e1b32b0c4fa5207f774b`
