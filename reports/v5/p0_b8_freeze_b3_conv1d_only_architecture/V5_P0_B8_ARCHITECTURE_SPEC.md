# Frozen P0 Architecture

**Selected:** B3 causal depthwise-separable Conv1D-only  
**Parameters:** 43,208  
**Weights frozen:** No — this is an architecture freeze.

## Interface
- PRIMARY58 `x`: `[B,16,58,32]`
- recovered topology-derived Boolean mask: `[B,16,10]`
- no second normalization
- no metadata, router IDs, coordinates, or graph message passing

## Encoder
- `Conv1d(58,64,1)`
- causal depthwise-separable residual blocks
- kernel 3, dilations `[1,2,4,8]`
- receptive field 31
- final causal encoded timestep

## Heads
- router representation: `Linear(74,64)->ReLU`
- role heads: `Linear(64,32)->ReLU->Linear(32,1)`
- graph pooling: mean + max
- graph encoder: `Linear(128,64)->ReLU`
- attack head: `Linear(64,1)`
- count head: `Linear(64,3)`

## Decision evidence
B7B issued `REJECT_B7A_AND_FREEZE_B3`.

## Contract hash
`86624dd842a57fe289f7d747151ad2aabf0fb5203d8fbc618f48ce3c91ef71b2`
