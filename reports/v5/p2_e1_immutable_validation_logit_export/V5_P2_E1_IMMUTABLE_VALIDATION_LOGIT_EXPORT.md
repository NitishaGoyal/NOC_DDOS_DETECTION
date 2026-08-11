# V5 P2 E1 Immutable Validation-Logit Export

## Frozen neural system

- Architecture: **Causal Depthwise-Separable Conv1D Temporal Encoder + Two-Layer GraphConv**
- Selected seed: **107**
- Selected epoch: **25**
- Checkpoint SHA-256: `82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc`
- Parameters: **59,785**

## Immutable archive

- Validation items: **12,528**
- Validation batches: **49**
- Inference backend: **CPU float32**
- AMP: **disabled**
- Deterministic algorithms: **enabled**
- First-batch exact repeated inference: **PASS**
- Raw logits exported: **true**
- Probabilities exported: **false**
- Threshold selection performed: **false**
- Decoder execution performed: **false**
- Stable item identities unique: **true**
- Archive manifest SHA-256: `ac7a38fd1f4b00b578254d9253418655aa78b6cc7a7a1d65f78ba15487b14604`

## Canonical label aliases

- `y_graph` ← loader `y_attack`
- `y_path` ← loader `y_attack_path`

## Exported arrays

- `graph_logits.npy` — `[12528]`, float32
- `count_logits.npy` — `[12528,4]`, float32
- `source_logits.npy` — `[12528,16]`, float32
- `transit_logits.npy` — `[12528,16]`, float32
- `victim_logits.npy` — `[12528,16]`, float32
- `path_logits.npy` — `[12528,16]`, float32
- Canonical graph/count/role labels
- Raw `role_mask`
- Validation and manifest row indices
- Static physical-port mask
- Stable item metadata JSONL

## Current state

- Validation predictions exported: **true**
- Raw thresholds frozen: **false**
- Decoder parameters frozen: **false**
- Decoder frozen: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Ready for one-shot test: **false**
- Next stage: **V5 P2 E2 raw-threshold freeze**
