# V5 P2 L1 XY Route Library and Unit Tests

## Frozen route library

- Topology: **4×4 cardinal 2D mesh**
- Router numbering: **row-major**
- Routing: **deterministic X-then-Y**
- Ordered source-victim routes: **240**
- Route IDs: **contiguous 0–239**
- Route table SHA-256: `19624d95a83114fba1e27e647aab59a0231c97d45053b598692ae62578b321b0`
- Python module SHA-256: `3ae860817ddef7cdbc0360057c51dc86908337b40b1df4c7df8fc2c8b82ea0d7`

## Mask semantics

- Source mask contains the source only.
- Transit mask excludes source and victim.
- Victim mask contains the victim only.
- Path mask equals source ∪ transit ∪ victim.
- Multi-route unions use bitwise OR.
- Shared victims, overlapping paths, and cross-route role overlap remain legal.

## Verification

- Unit tests run: **27**
- Result: **PASS**
- Test source SHA-256: `3d094019ad215af096dc9268fbdeba9ebdfd5e016dcc647193541825b9d8e45d`
- Test log SHA-256: `ec06252269dac62e1c877ff5e188e26dc0b75df6d4ba2892ade464219513475b`

The tests verify canonical coordinates, adjacency, all 48 directed physical
edges, all 240 X-then-Y routes, port direction legality, mask semantics,
multi-route union behavior, overlap cases, invalid-route rejection, count/source
consistency, and deterministic tie-breaking.

## Current state

- Structured decoder protocol locked: **true**
- Route library frozen: **true**
- Route-library unit tests passed: **true**
- Exact decoder implemented: **false**
- Beam decoder implemented: **false**
- Validation predictions exported: **false**
- Raw thresholds frozen: **false**
- Decoder frozen: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Ready for one-shot test: **false**
- Next stage: **V5 P2 L2 exact decoder implementation**
