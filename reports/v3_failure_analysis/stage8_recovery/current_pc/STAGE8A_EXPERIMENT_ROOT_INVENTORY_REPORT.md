# Stage 8A — V3 Experiment-Root Inventory

This stage performed a bounded, read-only inventory of the supplied experiment root.
It did not execute discovered files, follow symbolic links, load complete NumPy arrays,
or recursively search outside the experiment directory.

## Root

- Experiment root: `/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3`
- Files: **13**
- Directories: **0**
- Symbolic links recorded but not followed: **0**
- Total file size: **2823.721 MiB**
- Initial content verdict: **primarily_final_numpy_dataset_no_recognized_raw_outputs**

## Classification counts

- `1_raw_gem5_simulation_output`: **0**
- `2_intermediate_epoch_or_router_port_records`: **0**
- `3_processed_temporal_window_files`: **0**
- `4_final_numpy_graph_dataset`: **3**
- `5_labels_and_split_metadata`: **5**
- `6_builder_or_preprocessing_script`: **0**
- `7_logs_manifests_or_documentation`: **0**
- `8_unknown_or_unclassified`: **5**

## Interpretation boundary

Stage 8A establishes what artifacts exist and which files deserve targeted Stage 8B inspection.
It does not yet prove feature equations, normalization scope, clipping behaviour, port semantics,
window construction, label construction, or split provenance.

## Candidate raw artifacts

_None found._

## Candidate builder artifacts

_None found._

## Largest files

| relative_path | size_bytes | classification | npy_shape | npy_dtype |
|---|---|---|---|---|
| x.npy | 2872971392 | 4_final_numpy_graph_dataset | (233803, 16, 8, 24) | float32 |
| run_id.npy | 36473396 | 5_labels_and_split_metadata | (233803,) | <U39 |
| y_node.npy | 14963520 | 5_labels_and_split_metadata | (233803, 16) | float32 |
| active_cores.npy | 13093096 | 8_unknown_or_unclassified | (233803,) | <U14 |
| attackers.npy | 6546612 | 8_unknown_or_unclassified | (233803,) | <U7 |
| profile.npy | 6546612 | 8_unknown_or_unclassified | (233803,) | <U7 |
| split.npy | 4676188 | 5_labels_and_split_metadata | (233803,) | <U5 |
| seed.npy | 1870552 | 8_unknown_or_unclassified | (233803,) | <U2 |
| strength.npy | 1870552 | 8_unknown_or_unclassified | (233803,) | <U2 |
| end_epoch.npy | 935340 | 5_labels_and_split_metadata | (233803,) | int32 |
| y_graph.npy | 935340 | 5_labels_and_split_metadata | (233803,) | float32 |
| feature_cols.npy | 2240 | 4_final_numpy_graph_dataset | (24,) | <U22 |
| edge_index.npy | 1152 | 4_final_numpy_graph_dataset | (2, 64) | int64 |

## Unknown large files

_None found._

## Safety and integrity

- Traversal stayed lexically within root: **True**
- External symbolic link followed: **False**
- Source size/mtime/mode metadata unchanged: **True**
- NumPy header reads remained bounded: **True**
- Text-prefix reads remained bounded: **True**
- Walk/read errors recorded: **0**

## Next stage

Stage 8B should inspect only the ranked candidate raw and builder artifacts, then trace
the generation chain from simulator output to the final arrays.
