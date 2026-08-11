# V5-P3 EPLR-V1 governance

D6 freezes design only. It loads no model tensors, logits, training data,
validation data, or sealed-test data.

Sequence:
1. D7 route-library and synthetic unit tests.
2. D8 deterministic implementation and A-train-only calibration.
3. D9 one frozen Tranche-A validation exploratory replay.
4. D10 Raw/A0/old-A1/EPLR comparison and carry-forward decision.

D9 exploratory gate:
- graph predictions exactly equal Raw;
- active count predictions exactly equal Raw;
- source exact drop <= 1 percentage point;
- victim exact drop <= 1 percentage point;
- transit exact gain >= 5 percentage points;
- strict exact gain >= 5 percentage points;
- path does not materially degrade;
- every decoder failure is explicit and fail closed.

This branch does not block H1 quantization, shared 8x8 wrapper engineering, or
Tranche-B preparation. A-test remains sealed.
