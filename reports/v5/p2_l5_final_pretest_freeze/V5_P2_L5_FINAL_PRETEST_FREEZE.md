# V5 P2 L5 Final Pretest Freeze

## Frozen neural system

- Model: **P2TaskDGraphConvCount4**
- Selected checkpoint: **seed 107, epoch 25**
- Checkpoint SHA-256: `82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc`
- Model-source SHA-256: `ccdfcb74c1ddab7f20b98c922444873ad76c204ec4b5fd6827d52fb716f11fd9`

## Official blind-test result groups

1. **Raw neural outputs**
2. **A0 lightweight decoder**
3. **A1 exact structured decoder**

A2 is rejected and is not an official V5 blind-test decoder.

## Frozen decoders

- A0 graph threshold: **0.47174675035328983**
- A0 source threshold: **0.94960549299285935**
- A0 transit threshold: **0.85703332488359107**
- A0 victim threshold: **0.82601148026998117**
- A0 path threshold: **0.83393474660904676**
- A1 margin threshold: **8.7205320882398425**
- A1 semantic tolerance: **1e-12**
- A1 Decimal precision: **80 digits**

## Current boundary

- Metric contract frozen: **true**
- Concrete evaluator source frozen: **false**
- Test directory existence checked: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Ready for one-shot test: **false**
- Next stage: **V5 P2 preflight source and interface audit**

The next stage resolves and freezes the concrete evaluator and loader interfaces
without opening or enumerating the P2 test split.
