# Final V3 Architecture Freeze

**Status:** V3 ARCHITECTURE FROZEN

## Retained model

`Chrono-A1 Conv1D-TemporalGCN`

## Rejected ablations

- B1 scenario-balanced sampling
- C1 mean-plus-max graph readout

## Why C1 was rejected

C1 appeared promising on validation:

- graph F1 delta: `+0.016219`
- normal-run macro FPR delta: `-0.052460`

But it failed blind-test corroboration:

- graph F1 delta: `-0.038273`
- graph FPR delta: `+0.127189`
- normal-run macro FPR delta: `+0.127189`

Its weak-attack recall and node ranking improved slightly, but the graph-level
false-positive cost was unacceptable.

## Final decision

- No C1 multiseed training.
- No further V3 architecture or sampler experiments.
- Chrono-A1 is the final V3 model.
- The next phase is the V3.1 raw-trace and feature-semantics audit.

## Scientific interpretation

The remaining blind-test failure is not solved by scenario-balanced sampling
or by replacing mean pooling with mean-plus-max pooling. The evidence now points
more strongly toward feature semantics, representation ambiguity, and scenario
distribution mismatch.
