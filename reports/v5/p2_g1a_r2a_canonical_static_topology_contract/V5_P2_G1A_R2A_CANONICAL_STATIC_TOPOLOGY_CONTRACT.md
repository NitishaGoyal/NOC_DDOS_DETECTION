# V5 P2-G1A-R2A Canonical Static Topology Contract

- Status: **COMPLETE**

The P2 Conv1D dataset does not carry `edge_index` because the frozen B3 model did not use graph message passing. This stage therefore freezes a new explicit graph input rather than pretending to recover a missing sample field.

## Frozen topology

- 4×4 2D physical mesh.
- Router IDs 0–15.
- Row-major numbering: `router_id = row * 4 + column`.
- 24 physical bidirectional links.
- 48 directed edges.
- No physical self-loops.
- Lexicographically sorted directed edge order.

The generated topology is now the only authorized edge input for G1 graph baselines.

No model training, checkpoint loading, prediction-cache access, P2 test access, quantization, RTL generation, or Legal NoC decoder implementation was performed.
