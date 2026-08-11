# V5 P2 Prospective Recovery Protocol Amendment

## Decision

**ALLOW A NARROW PROSPECTIVE RECOVERY USING THE PRE-EXISTING B6 PAIRING RULE**

The original authorized execution remains a consumed, incomplete execution. It
aborted before test-directory enumeration, tensor access, model construction,
inference, decoding, prediction generation, or metrics.

## Sole amended interface

The recovery evaluator shall not require `split=test` rows in the frozen
train/validation A1-R2 manifest.

After a new recovery authorization is consumed, it shall instead apply the
pre-existing B6 rule frozen before the abort:

- exactly **138** `.pt` test tensors;
- exactly **69** complete ATTACK/CONTROL pairs;
- filenames `<pair_key>_ATTACK.pt` and `<pair_key>_CONTROL.pt`;
- lexicographic pair-key order;
- `common_length = min(T_attack,T_control)`;
- window **32**, stride **8**;
- pair-major, start-major, ATTACK then CONTROL.

## Unchanged contracts

- Model/checkpoint: **unchanged**
- Raw/A0/A1 output groups: **unchanged**
- A0 thresholds: **unchanged**
- A1 exact decoder and threshold: **unchanged**
- Metrics: **unchanged**
- A2: **rejected and not executed**
- One neural inference pass: **unchanged**

## Reporting requirement

A future recovery result must disclose the original pre-inference abort and
must be identified as a prospective recovery evaluation. It may not be
presented as the original one-shot execution.

## Current boundary

- Test path formed: **false**
- Test directory checked: **false**
- Test directory enumerated: **false**
- Test tensors opened: **false**
- New recovery authorization created: **false**
- Ready for recovery test: **false**
- Next stage: **V5 P2 recovery evaluator source freeze**

Amendment artifact SHA-256: `b2be725493603328bf7ae56a7ba0b8d6bdb3dc5aadc9ffdc52e3c3fc3cfde095`
