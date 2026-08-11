# V5 P2 Recovery Evaluator Source Freeze

## Status

- Status: **COMPLETE**
- Recovery evaluator SHA-256: `770ad8937427c32680c1cced7b3040bf6055662615c88edbf544e8805d9fe7c9`
- Recovery amendment bound: **true**
- Synthetic B6 pairing audit: **PASS**
- Source-only model/checkpoint/A1 integration: **PASS**

## Sole amended interface

The evaluator derives the 69 test pairs from the pre-existing B6 filename rule
instead of requiring test rows in the train/validation manifest.

- Expected `.pt` files: **138**
- Complete ATTACK/CONTROL pairs: **69**
- Pair order: **lexicographic**
- Window: **32**
- Stride: **8**
- Item order: **ATTACK then CONTROL per start**

## Unchanged science

- Model/checkpoint: **unchanged**
- Raw/A0/A1 output groups: **unchanged**
- A0 thresholds: **unchanged**
- A1 exact decoder and margin: **unchanged**
- Metrics: **unchanged**
- A2: **not executed**

## Current boundary

- New recovery authorization created: **false**
- Test path resolved: **false**
- Test directory checked: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Ready for recovery test: **false**
- Next stage: **V5 P2 recovery final no-data preflight and authorization**
