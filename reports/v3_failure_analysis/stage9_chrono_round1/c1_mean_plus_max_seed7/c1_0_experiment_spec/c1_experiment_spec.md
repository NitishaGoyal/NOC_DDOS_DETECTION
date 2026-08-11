# C1 Mean-Plus-Max Readout Specification

**Status:** SPECIFICATION FROZEN

## Control

`Chrono-A1 Conv1D-TemporalGCN`, ordinary shuffled training, seed 7.

## Sole permitted change

```text
A1: mean graph pooling
C1: concatenate mean and max graph pooling
```

The graph-head input width changes from `gcn_out` to `2 * gcn_out`.

## Critical prohibition

**The rejected B1 scenario-balanced sampler must not be imported or reused.**

## Validation gate

At least one primary improvement must pass and every mandatory safeguard must pass before blind-test transfer.

## V3 stopping rule

C1 is the final permitted V3 architecture intervention. After its verdict, V3 model development ends, except matched multi-seed confirmation if C1 is first judged promising.
