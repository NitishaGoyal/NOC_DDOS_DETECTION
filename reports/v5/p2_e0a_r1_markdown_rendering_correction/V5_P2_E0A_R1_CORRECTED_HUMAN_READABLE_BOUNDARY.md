# V5 P2 E0A Raw-Head / Structured-Decoder Boundary

## Selected neural architecture

- Selected full architecture: **Causal Depthwise-Separable Conv1D Temporal Encoder + Two-Layer GraphConv**
- Temporal frontend reference: **B3**
- Paper short name: **Temporal Conv1D-GraphConv**
- Selected seed: **107**
- Selected epoch: **25**
- Checkpoint SHA-256: `82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc`

The complete selected model must not be referred to only as B3. B3 is
the temporal frontend reference.

## Predeclared systems

- **A0:** independently thresholded raw neural baseline
- **A1:** exact deterministic legal-route decoder
- **A2:** hardware-oriented deterministic beam decoder

## Cache and reporting boundary

- A fresh E1 validation-logit archive is required.
- Legacy B4 and B6 caches may not be reused.
- A0, A1, and A2 metrics remain separate.
- No system, threshold, decoder, checkpoint, or variant may be selected after test access.

## Frozen order

```text
E0   threshold protocol lock — complete
E0A  raw-head / structured-decoder boundary — complete
L0   structured decoder protocol lock — next
L1   route-library generation and tests
L2   exact decoder implementation
L3   hardware beam decoder implementation
E1   immutable validation-logit export
E2   raw threshold selection and freeze
L4   decoder validation selection and freeze
L5   final pretest freeze bundle
PREFLIGHT
P2   one-shot blind evaluation of A0, A1, and A2
```

## Current state

- Raw/decoder boundary locked: true
- Structured decoder protocol locked: false
- Validation predictions exported: false
- Raw thresholds frozen: false
- Decoder frozen: false
- Test directory enumerated: false
- Test tensors deserialized: false
- Ready for one-shot test: false
