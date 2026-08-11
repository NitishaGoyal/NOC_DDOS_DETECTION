# V5 P2 Task-D Architecture Selection Freeze

- Status: **COMPLETE**
- Selected architecture: **GraphConv**
- Rejected architecture: **Conv1D**
- Selection basis: validation only
- Architecture selected: true
- Checkpoint selected: false
- Ready for one-shot test: false
- Threshold tuning performed: false
- Test directory enumerated: false
- Test tensors deserialized: false

## Selection rationale

- Selection-score delta: +0.007277 with GraphConv winning 4/5 seeds.
- Macro-role AP delta: +0.015536 with GraphConv winning 5/5 seeds.
- Victim AP delta: +0.036861 with GraphConv winning 5/5 seeds.
- Path AP delta: +0.019561 with GraphConv winning 5/5 seeds.
- Parameter increase: 38.16%.
- Linear-MAC increase: 2.41%.
- Graph-message scalar operations: 6,144.
