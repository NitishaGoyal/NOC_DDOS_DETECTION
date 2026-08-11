# EPLR-V2 Preserve-or-Passthrough design input

This is a D10 design input, not a frozen implementation.

## Proposed immutable outputs

- graph: Raw
- count: Raw
- source: Raw
- victim: Raw

## Feasible endpoint case

When exactly K legal XY routes preserve the Raw source and victim masks:

- select the best legal route tuple using train-frozen transit/path support;
- output route-derived transit and path union bitmaps;
- retain Raw source and victim bitmaps;
- report legal or low-support status.

## Infeasible endpoint case

Do not repair or reassign endpoints.

- pass through Raw source, transit, victim and path masks;
- report `RAW_ENDPOINTS_UNRESOLVED_PASSTHROUGH`;
- return no certified route IDs;
- do not claim that the passthrough masks form a legal route explanation.

## Governance

The D10 counterfactual metrics are post-hoc diagnostics only. They must not be
reported as a frozen A-validation result. EPLR-V2 requires a new semantic and
unit-test freeze, A-train-only implementation work, and independent future
validation. A-test remains sealed.
