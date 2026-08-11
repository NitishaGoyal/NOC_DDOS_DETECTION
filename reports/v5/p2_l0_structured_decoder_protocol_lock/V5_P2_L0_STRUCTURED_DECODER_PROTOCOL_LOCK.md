# V5 P2 L0 Structured-Decoder Protocol Lock

## Frozen neural system

- Selected full architecture: **Causal Depthwise-Separable Conv1D Temporal Encoder + Two-Layer GraphConv**
- Temporal frontend reference: **B3**
- Selected seed: **107**
- Selected epoch: **25**
- Checkpoint SHA-256: `82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc`

## Frozen routing semantics

- Topology: **4×4 cardinal 2D mesh**
- Router numbering: **row-major**
- Mapping: **router_id = row × 4 + column**
- Coordinate convention: **x = column, y = row**
- Routing: **deterministic X-then-Y**
- Ordered routes: **240**
- Route ordering: **lexicographic by (source, victim), excluding self-routes**
- Transit excludes source and victim.
- Path includes source, transit routers, and victim.
- Sources are distinct within one hypothesis.
- Shared victims and overlapping paths are legal.
- Role overlaps across different routes remain multi-label.

## Frozen structured score

```text
Q(H) =
    1.00 * graph_logit
  + 0.50 * log_softmax(count_logits)[K-1]
  + 1.00 * sum(source logits over source union)
  + 0.50 * sum(transit logits over transit union)
  + 0.75 * sum(victim logits over victim union)
  + 0.50 * sum(path logits over path union)
```

- Weights are inherited exactly from the frozen multitask loss.
- No score-weight validation grid is allowed.
- Each router contributes at most once to each role union.
- No route-length normalization or manually added route penalty is used.
- H0 has score zero and empty masks.
- The continuous decoder margin is `M = max_H Q(H)`.
- Attack is emitted only when `M >= tau`.

## A1 exact decoder

- Uses all 240 routes and evaluates K1–K4.
- No candidate pruning, beam approximation, timeout fallback, or nonzero optimality gap.
- L2 must freeze one deterministic single-threaded exact backend before E1.
- Failure to prove optimality is a hard failure.

## A2 hardware-oriented beam decoder

- Candidate routes per source: **B ∈ {1, 2, 4}**
- Beam width: **{4, 8, 16}**
- Uses the same route table and union score as A1.
- Adds only newly introduced role bits during beam expansion.
- Must meet all validation equivalence requirements or remain unauthorized.

## L4 selection rule

1. Maximize graph balanced accuracy.
2. Lower graph FPR.
3. Higher strict all-task exactness.
4. Higher macro role F1.
5. Threshold closest to zero.
6. Lower numeric threshold.

No scenario-specific, attack-specific, router-specific, or test-based selection is permitted.

## Current state

- Structured decoder protocol locked: **true**
- Route library frozen: **false**
- Exact decoder implemented: **false**
- Beam decoder implemented: **false**
- Validation predictions exported: **false**
- Raw thresholds frozen: **false**
- Decoder frozen: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Ready for one-shot test: **false**
- Next stage: **V5 P2 L1 route library and unit tests**
