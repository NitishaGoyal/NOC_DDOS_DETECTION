# P2 Post-Hoc Test-Informed Numerical-Recovery Evaluation

## Status

- Status: **COMPLETE**
- Classification: **post-hoc, test-informed numerical recovery**
- Blind/untouched result: **no**
- Test items: **11666**
- Test pairs: **69**
- Neural inference passes: **1**

## Frozen groups

| Group | Graph accuracy | Graph F1 | Graph FPR | Strict all-task exactness |
|---|---:|---:|---:|---:|
| Raw | `0.793845` | `0.739126` | `0.290940` | `0.603034` |
| A0 | `0.793845` | `0.739126` | `0.290940` | `0.620864` |
| Certified A1 | `0.793845` | `0.726113` | `0.264414` | `0.697240` |

## Mandatory disclosure

The original P2 blind one-shot evaluation did not produce a complete official
Raw/A0/A1 test result. The numerical policy, certified decoder, and staged
evaluator were repaired using validation and synthetic data. This completed
execution is therefore test-informed and post hoc. It is not an untouched
blind test and is not the original one-shot evaluation.

Raw and A0 were committed immutably before A1 began.

Artifact SHA-256: `6f07f6e81505fb7c15738e34026a20f1e4df3cfff25723c75fbf269a57167ddc`
