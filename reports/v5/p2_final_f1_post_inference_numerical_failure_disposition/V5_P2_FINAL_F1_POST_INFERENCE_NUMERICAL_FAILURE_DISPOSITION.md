# V5 P2 Final F1 Post-Inference Numerical-Failure Disposition

## Status

- Status: **COMPLETE**
- Classification: **POST-INFERENCE MID-A1 NUMERICAL-CERTIFICATION ABORT**
- Recovery authorization consumed: **true**
- Recovery rerun authorized: **false**
- Official complete test result obtained: **false**

## Established execution boundary

- Test access started: **true**
- Test directory enumerated: **true**
- Test tensors deserialized: **true**
- Model/checkpoint loaded: **true**
- Neural inference completed: **11666/11666**
- Neural inference batches: **92/92**
- A1 exact decoding completed: **false**
- Last preserved A1 progress lower bound: **6200/11666**

## Numerical failure

- Exception: `ExactDecoderError`
- Reported MIP gap: `2.4064574342771067e-14`
- Frozen float64 numerical-zero tolerance: `1.4210854715202004e-14`
- Excess above tolerance: `9.8537196275690632e-15`
- Gap/tolerance ratio: `1.693393875670834`

This disposition does not conclude that the model failed, the selected route was
illegal, or the MILP was materially suboptimal. Those questions belong to the
next frozen stage: a validation/synthetic numerical root-cause audit.

## Scientific disposition

The completed neural inference does not constitute an official Raw/A0/A1 test
result because the evaluator aborted before its final result-commit boundary.
Any future P2 execution must be separately disclosed as a **post hoc numerical-
recovery evaluation**.

## Safety boundary

- Test path formed by F1: **false**
- Test directory checked by F1: **false**
- Test directory enumerated by F1: **false**
- Test tensors deserialized by F1: **false**
- New authorization created: **false**

## Next stage

**V5 P2 decoder D0 numerical root-cause audit**

Artifact SHA-256: `0265de528df00fc84186688ad64621a1ac688a7aae65b2381f93fb8757c372e8`
