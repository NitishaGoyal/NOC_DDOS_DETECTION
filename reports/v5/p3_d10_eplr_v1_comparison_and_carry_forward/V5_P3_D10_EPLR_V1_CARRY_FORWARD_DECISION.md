# V5-P3 D10 EPLR-V1 carry-forward decision

## Classification

**V5-P3-1500-D70 Tranche-A Preliminary Diagnostic — EPLR-V1 Comparison and Carry-Forward Review**

## Decision

EPLR-V1 is **not carried forward as implemented**.

It satisfied the intended immutable graph/count contract and improved:

- transit exactness by `+0.05965463`;
- strict exactness by `+0.11887759`.

It failed the predeclared Pareto gate because it changed endpoint/path quality:

- source exactness by `-0.15698587`;
- victim exactness by `-0.13291470`;
- path exactness by `-0.13239142`.

The decoder and calibration must not be retuned using Tranche-A validation.

## Root-cause disposition

The endpoint-preserving branch is not the cause of the endpoint collapse:
source and victim outputs on preserved rows are bit-identical to Raw.

The damaging behavior is concentrated in:

- `ENDPOINT_REPAIR_APPLIED`;
- `NO_CERTIFIED_LEGAL_EXPLANATION`.

Therefore:

- legal route cleanup remains promising;
- endpoint reassignment is rejected for the next version;
- zero-mask fail-closed output must not be used as the prediction returned for
  metric evaluation when a legal explanation cannot be certified.

## Carry-forward

A new decoder version may be designed:

**EPLR-V2 Preserve-or-Passthrough**

- graph remains Raw;
- count remains Raw;
- source remains Raw;
- victim remains Raw;
- when Raw endpoints admit a legal explanation, legal routing may clean
  transit/path;
- otherwise all Raw role masks pass through unchanged and the decoder reports
  an unresolved diagnostic status;
- no endpoint-repair MILP is used;
- no second EPLR validation replay on Tranche-A is authorized.

A future confirmatory evaluation must use an independent future validation
boundary such as frozen Tranche-B validation or the final A+B protocol.
