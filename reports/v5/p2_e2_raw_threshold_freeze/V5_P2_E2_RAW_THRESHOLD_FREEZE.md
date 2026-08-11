# V5 P2 E2 Raw-Threshold Freeze

## Frozen validation-only thresholds

- Graph: **0.47174675035328983**
- Source: **0.94960549299285935**
- Transit: **0.85703332488359107**
- Victim: **0.82601148026998117**
- Path: **0.83393474660904676**

Graph maximizes validation balanced accuracy. Each role independently maximizes
flattened positive F1. All heads use the frozen ties: lower FPR, threshold
closest to 0.5, then lower numeric threshold. Predictions are positive when
`probability >= threshold`.

## Count and A0 rules

- Raw count prediction: **argmax over K1-K4**
- A0 graph-negative output: **count zero and empty role masks**
- Joint, per-scenario, per-seed and per-router threshold tuning: **disabled**
- Decoder execution during E2: **false**

## Selected validation metrics

- Graph balanced accuracy: **0.853961512**
- Graph FPR: **0.290716061**
- Raw macro role F1: **0.717829313**
- A0 macro role F1: **0.723024943**
- Raw attack-only count macro F1: **0.986728206**
- A0 all-item count macro F1: **0.749841829**
- A0 strict all-task exactness: **0.637691571**

## Current state

- Validation predictions exported: **true**
- Raw thresholds frozen: **true**
- Decoder parameters frozen: **false**
- Decoder frozen: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Ready for one-shot test: **false**
- Next stage: **V5 P2 L4 decoder validation selection and freeze**
