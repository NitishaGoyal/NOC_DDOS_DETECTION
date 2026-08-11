# P1 Protocol Boundary

1. Audit P1 independently.
2. Verify the frozen PRIMARY58/window32/stride8/mask interface.
3. Lock P1 train, validation, and test manifests.
4. Train the frozen B3 architecture from scratch on P1 train.
5. Use only P1 validation for early stopping, checkpoint selection,
   scheduling, and threshold calibration.
6. Evaluate the locked P1 test exactly once.
7. Train B2 as the required simple baseline.
8. Do not reopen architecture search from test results.
