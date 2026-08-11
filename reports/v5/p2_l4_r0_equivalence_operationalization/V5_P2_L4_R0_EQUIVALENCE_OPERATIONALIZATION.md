# V5 P2 L4-R0 Decoder Equivalence Operationalization

Frozen prospectively before inspecting A1 or A2 validation-decoder results.

A2 eligibility requires graph-decision agreement with A1 of at least 0.999,
full-hypothesis agreement of at least 0.99, and no degradation greater than
0.001 absolute in graph balanced accuracy, all-item count macro F1, macro role
F1, or strict all-task exactness. Graph FPR may increase by at most 0.001.

Full-hypothesis agreement compares the final attack decision, count, sorted
route-ID tuple, and source/transit/victim/path union masks. H0 uses count zero,
route IDs -1, and zero masks.

Eligible configurations are ordered by lower B, lower beam width, higher
full-hypothesis agreement, then higher graph-decision agreement. If none is
eligible, L4 is HOLD and does not authorize L5 or blind testing.
