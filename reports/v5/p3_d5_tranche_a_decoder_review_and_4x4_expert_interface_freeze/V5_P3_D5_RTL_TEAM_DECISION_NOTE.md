# V5-P3 D5 decision for the RTL team

## Frozen neural expert interface

- Input: `x[16][70][32]`
- Physical-port mask: `[16][10]`
- Fixed topology: 4x4 mesh, 48 directed edges
- Current model parameters: 60,553
- Output: 69 raw neural logits
- Current checkpoint: Tranche-A epoch 14
- H0 handoff SHA-256: `431298f6870489b8b77a1ca0fae774aff1604ef2b6e11155d985a818a21689f0`

## Hardware/software boundary

The accelerator must produce the 69 raw logits. The exact structured MILP
decoder remains software-side and is not part of the synthesizable
Conv1D-GraphConv engine.

The current primary hardware graph-detection reference remains the raw graph
logit with the preliminary 0.5 validation policy. The final A+B model may
require a newly frozen threshold, but the 69-logit interface will not change.

## Decoder review

- Raw strict exact: `0.63211426`
- A1 strict exact: `0.75286734`
- A1 strict gain: `0.12075308`
- Raw graph accuracy: `0.80213518`
- A1 graph accuracy: `0.79203636`
- Raw graph FPR: `0.21302659`
- A1 graph FPR: `0.24489593`

A1 is useful for software-side structured localization because it substantially
improves strict consistency. It does not replace the raw neural graph detector,
because graph accuracy decreases and false-positive rate increases.

## Current implementation boundary

RTL may proceed with memory planning, scheduling, testbench construction,
quantization integration, Conv1D, GraphConv, pooling and output-head datapaths.

The current epoch-14 weights are a preliminary engineering reference. A future
fresh A+B model will preserve the same tensor and output interface.
