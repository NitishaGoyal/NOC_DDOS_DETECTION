# V5-P3 EPLR-V2 frozen software handover

## Classification

**V5-P3-1500-D70 Tranche-A Preliminary Diagnostic — EPLR-V2 Frozen Software Handover**

This archive contains the frozen EPLR-V2 software decoder and its contracts.
It does not contain a successful EPLR-V2 validation result.

## Included implementation

- `v5_p3_eplr_v2.py`
- `v5_p3_eplr_v2_calibration.json`
- `v5_p3_eplr_v1.py` as the endpoint-preserving route-selection dependency
- `v5_xy_route_library.py`

## Frozen semantics

- graph, count, source and victim remain Raw;
- legal endpoint sets may receive route-derived transit/path unions;
- infeasible endpoint sets pass through all Raw role masks;
- unresolved rows carry no route certificate;
- endpoint repair is absent.

## Validation state

EPLR-V2 is waiting for an independent boundary.

Permitted:
- frozen Tranche-B validation;
- future fresh A+B validation.

Not permitted:
- another Tranche-A validation replay;
- A-test access;
- using the D10 post-hoc counterfactual as a frozen candidate claim.

## Hardware boundary

EPLR-V2 remains software-side. The RTL accelerator interface remains the
existing 69 raw logits.
