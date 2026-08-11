# V5 P2-G0 Graph-Baseline and RTL-Handoff Protocol

## Frozen decision

Pause Legal NoC decoder implementation after C2A and run a time-boxed, validation-only graph-baseline branch first.

## Scientific questions

1. Can GraphConv reproduce strong binary source localization?
2. Does graph aggregation improve direct graph detection?
3. Does graph aggregation blur source/transit/victim/path roles?
4. Was the earlier localization problem caused by vanilla GCN, task formulation, data/labels, or graph learning generally?

## Required task order

`G1 source-only -> G2 graph detection -> G3 role-aware/full multitask -> G4 five-seed selection -> G5 RTL selection contract -> G6 RTL handoff -> return to L0 Legal XY decoder contract`

## Mandatory operators

- Conv1D-only reference.
- Conv1D + GCNConv.
- Conv1D + GraphConv.
- GAT is screening-only and requires a material validation or stability advantage for promotion.

## Fairness

All variants use the same P2 train/validation pair manifests, PRIMARY58 interface, physical-port mask, temporal encoder, graph width, layer count, seeds, optimizer, scheduler, early stopping, checkpoint-selection metric, and threshold-selection policy for a given task.

## Test boundary

- No P2 test inference.
- No P2 test tensor access.
- No B6-cache architecture comparison.
- First clean blind graph-model comparison occurs on V6.

## RTL direction

The preferred candidate is Conv1D + GraphConv source-only, but it is not selected yet. It must pass validation, stability, and hardware-cost gates. Existing B3 Conv1D-count4 is the fallback temporal reference.

## Immediate next stage

`V5_P2_G1_SOURCE_ONLY_GRAPH_BASELINES`
