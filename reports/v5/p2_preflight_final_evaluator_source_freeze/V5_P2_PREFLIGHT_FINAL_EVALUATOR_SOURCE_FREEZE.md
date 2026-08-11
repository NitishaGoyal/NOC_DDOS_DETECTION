# V5 P2 Final Evaluator Source Freeze

## Frozen evaluator

- Source: `/home/zira/research/projects/GNN-2d/scripts/v5/p2/evaluate_v5_p2_final_raw_a0_a1_one_shot.py`
- SHA-256: `f6f061177b1621e3e7d8b3cbb32a401f83cdbf0488235125c62f3be17501ee44`
- Embedded test loader: **frozen**
- Source-only model/checkpoint forward test: **PASS**
- Source-only exact-decoder integration test: **PASS**

## Official result groups

1. **Raw neural outputs**
2. **A0 lightweight decoder**
3. **A1 exact structured decoder**

A2 is not imported or executed.

## Execution contract

- Neural inference passes over test data: **exactly one**
- Device: **CPU**
- Precision: **float32**
- AMP: **disabled**
- DataLoader shuffle: **false**
- Test loader construction occurs only after authorization-token consumption.
- Any failure after token consumption is irreversible and does not authorize a rerun.

## Current boundary

- Authorization contract created: **false**
- Authorization token created: **false**
- Test path resolved: **false**
- Test directory existence checked: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Ready for one-shot test: **false**
- Next stage: **V5 P2 final no-data preflight and authorization**
