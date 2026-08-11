# V5 P2-C2A Legal NoC Decoder Amendment

## Status

C2 remains authoritative and unchanged. This append-only amendment adds an orthogonal structured-decoder study.

## Frozen pipeline

`B3 causal Conv1D-count4 -> raw graph/count/source/transit/victim/path scores -> Legal NoC structured decoder -> coherent attack hypothesis`

## Initial scope

- 4x4 2D mesh with deterministic XY routing.
- K1-K4 sources.
- One victim assignment per source.
- Shared victims and overlapping legal routes are allowed.
- Source/transit/victim roles may overlap.
- One source targeting multiple victims is deferred.
- The four-class count head remains unchanged.

## Graph decision

The graph head is not discarded and is not the only hard gate. The final decision must compare the normal hypothesis against the best legal structured attack hypothesis using graph and node/route evidence jointly.

## Evaluation boundary

- P2 B6 remains the official blind result.
- P2 decoder selection uses B4 validation outputs only.
- Frozen-decoder analysis on B6 is post-hoc exploratory only.
- V6 provides the first clean blind-test result for model plus decoder.

## Implementation gate

Decoder implementation is not yet authorized. L0 must first freeze label semantics, hypothesis construction, legality constraints, scoring, count handling, tie-breaking, validation grid, and exact-versus-hardware equivalence targets.

## Next stage

`V5_P2_L0_LEGAL_XY_DECODER_CONTRACT`
