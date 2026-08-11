# V4 A4a Final Scientific Closure

## Final status

**Hypothesis outcome:** `FAILED_TO_RECOVER_CARDINALITY_GAP`

A4a-2B evaluated all 12 retained
validation checkpoints. No checkpoint passed every frozen gate, no checkpoint
was selected, and development-test transfer was not authorized.

## Frozen A3 reference

- A3-H32 validation attack exact localization: 78.78%
- A3-H32 oracle-count exact localization: 91.37%
- A3-H32 oracle gap: 12.59 percentage points
- Required A4a validation exact localization: 80.78%

## Diagnostic best A4a checkpoint

- Epoch: 24
- Temporal attack exact localization: 55.58%
- Stable attack exact point rate: 55.21%
- Attacker-set precision: 98.27%
- Attacker-set recall: 66.10%
- Candidate coverage: 86.13%
- Persistent normal-run false-isolation fraction: 0.00%
- Gain over A3-H32: -23.20 percentage points

## Raw checkpoint evidence

- Membership precision: 92.14%
- Membership recall: 57.07%
- Membership F1: 70.49%
- Membership exact fraction: 60.93%
- Cardinality accuracy: 62.66%
- All-NULL proposal fraction: 40.15%
- Duplicate proposal fraction: 12.80%

## Interpretation

The structured slot decoder remained precision-oriented but under-predicted attacker membership and cardinality. The diagnostic checkpoint overused NULL, achieved insufficient membership recall, failed the candidate-coverage gate, and remained far below the frozen A3-H32 exact-localization reference.

A4a retained a useful precision/safety operating point, but it did not recover
the A3.12 cardinality gap. The remaining failure is consistent with missing
source/transit/victim and route-transition semantics in V4 rather than a small
checkpoint-selection error.

## Final boundary

No A4a development-test transfer will be run. Further V4 NULL, cardinality,
loss-weight, or decoder searches are closed. The next stage is V5 dataset
contract freeze and instrumentation-pilot validation.

## Audit boundary

- Dataset samples loaded: no
- Model inference performed: no
- Threshold search performed: no
- Test loader constructed: no
- Test evaluated: no
- Development test accessed: no
