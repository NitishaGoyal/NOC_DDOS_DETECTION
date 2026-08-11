# V5 P2 A3 Pair-Aligned PRIMARY58 Loader Contract

## Authorized loader

`V5P2PairAlignedPrimary58Dataset`

Authorized splits:

```text
train
validation
```

The constructor rejects `test`.

## Windowing

```text
window = 32
stride = 8
common_length = min(T_attack, T_control)
ordering = pair, start, ATTACK, CONTROL
```

Both pair members use identical starts and final-epoch targets.

## Returned item

```text
x                  float32 [16,58,32]
physical_port_mask bool    [16,10]
y_attack           float32 scalar
y_attacker_count   int64   scalar
y_source           float32 [16]
y_transit          float32 [16]
y_victim           float32 [16]
y_attack_path      float32 [16]
role_mask          uint8/bool [16]
```

Only `x` and `physical_port_mask` are model inputs. The rest are targets or
loss masks.

The loader returns no identifiers, metadata, coordinates, edge index, lengths,
or window positions.

## Normalization

`x` is an exact slice of the stored train-standardized tensor. The loader does
not normalize, standardize, inverse-transform, or otherwise alter feature
values.

## Counts

```text
train pairs          = 415
validation pairs     = 74
total aligned items  = 82694
```

## Contract SHA-256

`35c3ecf8d48ed763341e45bad3a1730df4762f567d2d3d35380cbd72fbc9312d`

## Test boundary

No P2 test directory was enumerated and no P2 test tensor was opened.
