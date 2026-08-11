# V5 P2 Task-D Full-Multitask Package

This package implements the frozen joint G1/G2/G3 review and the prescribed
Task-D comparison using **train and validation only**.

## Scientific status

GraphConv is frozen as the single Task-D reviewer challenger because it is the
strongest non-attention operator across G1-G3. The joint review also records
that it does **not** satisfy every G0 case-2 material-gain threshold; therefore
B3 Conv1D remains the main lightweight path until the full Task-D comparison is
complete. No architecture is selected by this package.

## Matrix

- Candidates: `conv1d`, `graphconv`
- Seeds: `107,117,127,137,147`
- Runs: 10, serial CUDA, AMP disabled
- Inputs: `[B,16,58,32]` plus `[B,16,10]` physical-port mask
- Outputs: graph attack, count K1-K4, source/transit/victim/path `[16]`
- Loss, optimizer, scheduler, early stopping, checkpoint score and tie-breakers:
  exact frozen P2-B1 full-multitask protocol
- Threshold tuning during training: forbidden
- P2 test access: forbidden

## Architecture fairness

For a matched seed, both candidates instantiate the same frozen B3 reference.
The GraphConv challenger deep-copies every common temporal, node, graph-readout
and task-head parameter before adding two `GraphConv(64,64,aggr="add")` layers.
The preflight and final aggregation verify identical common initialization
hashes for each paired seed.

## Required order

1. Freeze joint G1/G2/G3 review.
2. Run Task-D preflight.
3. Run the serial 10-run matrix.
4. Aggregate results.
5. Proceed to G4 validation-only stability and architecture selection.

The package does not enumerate, deserialize, infer on, or tune against P2 test.
