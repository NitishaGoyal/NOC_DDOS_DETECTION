# V5 P2 L3 Hardware-Oriented Beam Decoder

## Frozen grid

- Per-source candidates: **B ∈ {1, 2, 4}**
- Beam widths: **{4, 8, 16}**
- Counts: **K1–K4**
- Candidate sources: **all 16**
- Fixed-point format: **deferred**
- RTL implemented: **false**

## Deterministic behavior

Each source retains its top-B legal routes using the frozen single-route role
score. Candidate ties use lower route ID. Beam states enforce distinct sources
and strictly increasing route IDs. Partial states are ordered by higher union
role score and then lower route tuple. Newly introduced role bits are counted
once; graph and count evidence are added only to complete K-route states.

## Verification

- Tests run: **12**
- Result: **PASS**
- All nine B/width configurations: **PASS**
- Known K1 beam-versus-exact case: **PASS**
- Known K2 beam-versus-exact case: **PASS**
- Incremental union-score recomputation: **PASS**
- Repeated determinism: **PASS**
- Module SHA-256: `07a0a98dbe1395f92595cab11cbca6f17e376d26ef68604c19baab333319f3aa`
- Route-table SHA-256: `19624d95a83114fba1e27e647aab59a0231c97d45053b598692ae62578b321b0`
- Exact-decoder SHA-256: `0d6ae843a0503cf6aa347078dda50e570f69cbbffdaa7ee1e4b2dbfd3b7978bf`

## Boundary

L3 freezes the software hardware-oriented implementation and search grid. It
does not select B or beam width, claim validation agreement, freeze fixed-point
widths, or implement RTL.

## Current state

- Beam decoder implemented: **true**
- Beam grid frozen: **true**
- Validation predictions exported: **false**
- Raw thresholds frozen: **false**
- Decoder frozen: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Ready for one-shot test: **false**
- Next stage: **V5 P2 E1 immutable validation-logit export**
