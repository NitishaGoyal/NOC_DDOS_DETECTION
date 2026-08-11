# V5 P0 Final Result and P2 Handoff

## Official P0 model

- B3 causal depthwise-separable Conv1D-only
- 43,208 parameters
- selected seed 117, epoch 101
- checkpoint SHA-256:
  `118c58ec6f9e62ce2a4ee81b4c4dbad75d53622ccb1f33ffa67db14230b59acc`

## Frozen thresholds

```text
attack  = 0.1
source  = 0.15
transit = 0.45
victim  = 0.8
path    = 0.15
```

## Official locked P0 test

```text
graph accuracy          = 0.8997
graph balanced accuracy = 0.9076
graph precision         = 0.8303
graph recall            = 0.9580
graph F1                = 0.8896
graph FPR               = 0.1429

count macro F1          = 0.9098
source F1               = 0.9056
transit F1              = 0.9375
victim F1               = 0.9377
path F1                 = 0.9526
all-task exact          = 0.5973
```

The P0 test was accessed exactly once. It must not be used again for model,
checkpoint, or threshold selection.

## Route correction

P1 is skipped and was not executed. The project proceeds directly to:

`V5_P2_INDEPENDENT_DATASET_AUDIT`

## Main P0 limitation

The model retained high attack recall but generated 56 false positives and 12
false negatives on the locked test. The primary P2 question is whether the
larger/more representative P2 dataset reduces control-traffic false positives
without sacrificing localization.

## P2 handoff hash

`8c5bf457408ac91b9be43e2dbf366a641fa3af1665b8347bf4722cd7da97e276`
