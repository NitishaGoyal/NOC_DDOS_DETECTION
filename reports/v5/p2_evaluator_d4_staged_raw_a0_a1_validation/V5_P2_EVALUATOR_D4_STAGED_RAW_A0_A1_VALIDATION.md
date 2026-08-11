# V5 P2 D4 Staged Raw/A0/A1 Evaluator Validation

## Status

- Status: **COMPLETE**
- Validation items: **12528**
- Pipeline order:
  **E1 archive → E2 Raw → E3 A0 → E4 certified A1**
- Synthetic A1-failure isolation: **PASS**
- Deterministic exact recomputation: **PASS**

## Persistence proof

Raw and A0 outputs were committed and hash-locked before A1 began. A synthetic
A1 technical failure was then injected. Every protected E1/E2/E3 hash remained
unchanged. The real E4 stage subsequently consumed only the immutable D3
certified-output archive.

Therefore:

```text
A1 technical failure does not erase or invalidate Raw/A0.
```

## Validation metrics

### Raw neural

- Graph accuracy: `0.794141`
- Graph F1: `0.739942`
- Graph FPR: `0.290716`
- Macro role F1: `0.717829`
- Joint role exact match:
  `0.614863`
- Strict all-task exactness:
  `0.612947`

### A0 lightweight

- Graph accuracy: `0.794141`
- Graph F1: `0.739942`
- Graph FPR: `0.290716`
- Macro role F1: `0.723025`
- Joint role exact match:
  `0.639607`
- Strict all-task exactness:
  `0.637692`

### A1 certified structured

- Graph accuracy: `0.802123`
- Graph F1: `0.738640`
- Graph FPR: `0.260673`
- Macro role F1: `0.729831`
- Joint role exact match:
  `0.731003`
- Strict all-task exactness:
  `0.731003`
- Route-legality rate: `1.000000`

These are validation metrics, not test results. No threshold, model, checkpoint,
or decoder was selected or changed.

## Safety boundary

- Test path formed: **false**
- Test directory checked/enumerated: **false**
- Test tensors deserialized: **false**
- Model/checkpoint loaded: **false**
- Neural inference performed: **false**
- Threshold selection performed: **false**
- Decoder execution performed: **false**
- Evaluation authorization created: **false**

## Next stage

**V5 P2 evaluator D5 finalized-architecture E1 integration and validation**

Artifact SHA-256: `0be3b06a31b30fabda6e7f6f5582be9cf4bfde105a4d01bf893dd4bcc0545a3b`
