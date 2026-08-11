# V5-P2 Canonical Neural + Legal-Router Preservation

This directory is the immutable reference system preserved before retrospective
R0/R1 model-improvement work.

## Canonical neural model

- Architecture: **Causal Depthwise-Separable Conv1D Temporal Encoder + Two-Layer GraphConv**
- Parameters: **59,785**
- Checkpoint: **seed 107, epoch 25**
- Checkpoint SHA-256: `82314baedb4bf3969842abf9076679682369daae2e57db07d4b4aeaa8623f7cc`

## Canonical legal-route system

- Certified decoder policy: `V5P2-D1-6ce8cc7c76b8d2cff367631077d65f20`
- Route library SHA-256: `3ae860817ddef7cdbc0360057c51dc86908337b40b1df4c7df8fc2c8b82ea0d7`
- Certified decoder SHA-256: `30309376cb0990a0f482045e2dac413d4370729cf94f9e556f895af2c5c8c71d`

## Final P2 evidence

The preserved D8 result is the completed:

> **P2 Post-Hoc Test-Informed Numerical-Recovery Evaluation**

D8 artifact SHA-256:

`6f07f6e81505fb7c15738e34026a20f1e4df3cfff25723c75fbf269a57167ddc`

The P2 test result is preserved for diagnosis and reporting only. It must not be
used to select R0/R1 or later model variants.

## Retrospective boundary

All new models, checkpoints, logs, reports, and metrics must be written outside
this directory under dedicated retrospective paths. No file in this
preservation directory may be edited or replaced.
