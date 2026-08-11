# V5 P2 E0 Validation Threshold Protocol Lock

## Frozen neural candidate

- Technical name: **Causal Depthwise-Separable Conv1D Temporal Encoder + Two-Layer GraphConv (B3)**
- Paper short name: **Temporal Conv1D-GraphConv**
- Selected seed: **107**
- Selected epoch: **25**
- Checkpoint SHA-256: `82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc`

## Graph threshold

- Data: validation split only
- Objective: maximize balanced accuracy
- Tie-break 1: lower false-positive rate
- Tie-break 2: threshold closest to 0.5
- Tie-break 3: lower numeric threshold

## Role thresholds

- Heads: source, transit, victim, and path
- Thresholds selected independently
- Objective: maximize positive-class F1 over flattened item-router entries
- Tie-break 1: lower false-positive rate
- Tie-break 2: threshold closest to 0.5
- Tie-break 3: lower numeric threshold

## Candidate thresholds

- Apply sigmoid to frozen validation logits
- Include thresholds 0.0, 0.5, and 1.0
- Include midpoints between adjacent unique probabilities
- Prediction is positive when probability is greater than or equal to the threshold
- No manual threshold insertion or visual selection

## Attacker count

- Use argmax over classes 1, 2, 3, and 4
- No attacker-count threshold
- Graph-negative to count-zero mapping is deferred to the deterministic decoder

## Frozen constraints

- No joint five-threshold optimization
- No per-scenario, per-attack, per-router, or per-seed thresholds
- No architecture or checkpoint changes
- No retraining
- No threshold reselection after blind-test evaluation
- Raw neural metrics and decoder metrics must be reported separately

## Current state

- Architecture selected: true
- Checkpoint selected: true
- Threshold protocol locked: true
- Validation predictions exported: false
- Thresholds frozen: false
- Decoder frozen: false
- Test directory enumerated: false
- Test tensors deserialized: false
- Ready for one-shot test: false
