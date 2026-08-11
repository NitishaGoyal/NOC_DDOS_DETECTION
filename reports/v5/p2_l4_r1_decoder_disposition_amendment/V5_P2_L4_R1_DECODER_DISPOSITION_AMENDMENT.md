# V5 P2 L4-R1 Decoder Disposition Amendment

## Frozen outcome

L4 completed with **0 / 9** A2 configurations satisfying the prospectively
frozen equivalence and metric-loss requirements. No requirement is relaxed and
no A2 configuration is selected after observing validation performance.

## Official V5 blind-test output groups

1. **Raw neural outputs** — unmodified model-output reference.
2. **A0** — frozen independent-threshold lightweight/deployment baseline.
3. **A1** — frozen exact legal XY structured software reference.

A2 remains a validation-only rejected approximation study and is not an
official blind-test decoder.

## Frozen A0 thresholds

- Graph: **0.47174675035328983**
- Source: **0.94960549299285935**
- Transit: **0.85703332488359107**
- Victim: **0.82601148026998117**
- Path: **0.83393474660904676**

## Frozen A1 rule

- Margin threshold: **8.7205320882398425**
- Semantic tie tolerance: **1e-12**
- High-precision score evaluation: **80 Decimal digits**
- Route library: **240 ordered non-self deterministic X-then-Y routes**
- Validation items certified: **12,528 / 12,528**

## Prohibited actions

- Relaxing the A2 gate after observing L4: **prohibited**
- Selecting an A2 configuration post hoc: **prohibited**
- Running A2 as an official V5 blind-test decoder: **prohibited**
- Additional V5 decoder tuning: **prohibited**
- Test-driven corrections: **prohibited**

## Current state

- Decoder disposition frozen: **true**
- A2 rejected: **true**
- Official test decoders: **A0, A1**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Ready for one-shot test: **false**
- Next stage: **V5 P2 L5 final pretest freeze**
