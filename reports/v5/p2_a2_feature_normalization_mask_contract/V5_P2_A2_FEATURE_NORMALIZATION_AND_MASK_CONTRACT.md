# V5 P2 Feature, Normalization, and Mask Contract

## Decision

P2 is compatible with the frozen P0 `PRIMARY58` learned-input contract.

```text
stored x              = [T,16,81]
selected temporal x   = indices 0-29, 31-55, 62, 64, 65
selected feature count= 58
window                = 32
model temporal input  = [B,16,58,32]
physical mask         = topology-derived Boolean [B,16,10]
```

## Normalization

The stored tensors already use:

```text
log1p_then_train_standardize
fit split = train
```

A second normalization pass is forbidden.

## Port-valid channels

Stored indices 71-80 are not trusted as raw model-mask inputs. The physical
mask is reconstructed from `topology.pt` and passed separately.

Observed maximum inverse-transform recovery error on deterministic non-test
samples:

```text
1
```

## Frozen B3 interface

The frozen B3 architecture has no graph message passing. `edge_index` and
router coordinates are audit/bookkeeping artifacts only and are not model
inputs.

## Loader status

The supplied `dataset_loader.py` is not authorized for training. A new loader
must enforce the A1-R2 pair-aligned window contract, select PRIMARY58, return
the topology-derived mask, and quarantine all provenance fields.

## Contract hash

`9b8ae700dfcd232725408ee35c018ec7fe48ee3cc1ec3de84a2477eae8ed3755`

## Test boundary

No P2 test-directory enumeration or test-tensor access occurred.
