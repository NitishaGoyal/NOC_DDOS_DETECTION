# V5 P2 Pair-Aligned Window Contract

## Decision

Use a common-prefix window index for every TRAIN and VALIDATION matched pair.

```text
common_length = min(T_attack, T_control)
window = 32
stride = 8
starts = 0, 8, 16, ... while start + 32 <= common_length
```

Both ATTACK and CONTROL members receive exactly the same ordered starts.

## Audit result

```text
non-test pairs                         = 489
pairs with unequal epoch lengths       = 77
pairs with unequal native window count = 14
native windows                         = 82708
aligned windows                        = 82694
windows removed                        = 14
removed fraction                       = 0.00016927
```

Only 14 windows are removed, so the leakage control costs approximately
0.0169% of the native non-test windows.

## Security rule

Run length, pair identity, mode, category, window position, terminal distance,
filenames, IDs, and all derived provenance values are bookkeeping only and
must never be supplied to the model.

## Source immutability

Original `.pt` tensors remain unchanged. Alignment is applied by the loader
when constructing eligible windows.

## Test boundary

No P2 test directory enumeration or test tensor access occurred in this stage.

## Contract hash

`857b679c40c512181ec5b060d7bc1f66fb9d19edc65b6dbc593ad68554106420`
