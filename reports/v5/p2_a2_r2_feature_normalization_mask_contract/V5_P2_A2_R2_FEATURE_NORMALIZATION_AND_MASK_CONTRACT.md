# V5 P2 A2-R2 Feature, Normalization, and Mask Contract

## Historical resolution

The original A2 HOLD was caused by the wrong directional interpretation of
`router_coordinates`, not by corrupted port-valid channels.

A2-R1 proved a unique zero-error convention:

```text
vertical axis   = coord[0]
horizontal axis = coord[1]
north           = increasing coord[0]
east            = increasing coord[1]
```

Across 1,355,200 compared values:

```text
distance to nearest Boolean = 0
temporal variation          = 0
input/output difference     = 0
mask mismatch count         = 0
```

## Frozen learned-input contract

```text
stored x            = [T,16,81]
selected PRIMARY58  = 0-29, 31-55, 62, 64, 65
window              = 32
model temporal x    = [B,16,58,32]
topology mask       = Boolean [B,16,10]
```

The stored tensor is already `log1p` transformed and standardized using
train-only statistics. A second normalization pass is forbidden.

## Correct physical-port rules

```text
local = true
north = vertical < vertical_max
east  = horizontal < horizontal_max
south = vertical > vertical_min
west  = horizontal > horizontal_min
```

The same five channels are used for input and output physical validity.

## Frozen B3

B3 has no graph message passing. `edge_index` and router coordinates remain
audit/bookkeeping data and are not model inputs.

## Loader requirement

The supplied reference loader is not authorized. A replacement must enforce:

- A1-R2 pair-aligned common-prefix windows;
- PRIMARY58 selection;
- corrected topology-derived Boolean mask;
- no second normalization;
- no metadata or provenance in returned model inputs.

## Contract SHA-256

`4c9493e5a0b9d2624af550953a728b8b9f2c983eaeaca9aa52e5696928b9ee54`

## Test boundary

No P2 test directory was enumerated and no test tensor was opened.
