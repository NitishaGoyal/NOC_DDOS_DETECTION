# V5 P2 L4-R1-R0 E2 Threshold-Schema Resolution

The initial L4-R1 freezer stopped before writing any disposition artifact
because it treated E2 threshold-selection objects as scalar thresholds.

The frozen E2 schema is:

- `graph_threshold.selected_threshold`
- `role_thresholds.source.selected_threshold`
- `role_thresholds.transit.selected_threshold`
- `role_thresholds.victim.selected_threshold`
- `role_thresholds.path.selected_threshold`

The exact E2 artifact hash and all five previously frozen numerical thresholds
were verified. No scientific rule, threshold, decoder, model, validation
prediction, or test data was changed or inspected.
