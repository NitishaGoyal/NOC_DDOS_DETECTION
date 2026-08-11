# Stage 8D Final V3 Consistency Report

- Generated: `2026-07-18T18:28:36.871179+00:00`
- Dataset: `/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3`
- Verdict: **PASS WITH DOCUMENTED LIMITATIONS**
- Checks passed: **30/30**
- Hard failures: **0**
- Warning-level findings: **0**

## Scope

This audit checks whether the unpacked V3 arrays obey the recovered builder logic. The main tensor was opened read-only with NumPy memory mapping and scanned in chunks.

## Expected structure

- Samples: `233803`
- Runs: `71`
- Samples per run: `3293`
- End epochs: `7..3299`
- Tensor nodes/time/features: `16 × 8 × 24`

## Validation results

| Status | Severity | Check | Details |
|---|---|---|---|
| PASS | hard | `x_shape` | observed=(233803, 16, 8, 24) expected=(233803, 16, 8, 24) |
| PASS | warning | `x_dtype` | observed=float32 expected=float32 |
| PASS | hard | `x_memory_mapped` | type=memmap |
| PASS | hard | `sample_array_alignment` | y_graph=(233803,); y_node=(233803, 16); run_id=(233803,); end_epoch=(233803,); split=(233803,); active_cores=(233803,); profile=(233803,); seed=(233803,); strength=(233803,); attackers=(233803,) |
| PASS | hard | `y_node_shape` | observed=(233803, 16) |
| PASS | hard | `feature_columns` | observed_count=24 expected_count=24 |
| PASS | hard | `unique_run_count` | observed=71 expected=71 |
| PASS | hard | `run_contiguity` | segments=71 unique_runs=71 |
| PASS | hard | `no_repeated_run_segments` | repeated=[] |
| PASS | hard | `samples_per_run` | expected=3293; failures=0 |
| PASS | hard | `end_epoch_sequence` | expected=7..3299; failures=0 |
| PASS | hard | `run_level_split_isolation` | run_counts_by_split={'train': 45, 'val': 13, 'test': 13} |
| PASS | hard | `graph_labels_constant_and_binary` | Every run must have one graph label in {0,1}. |
| PASS | hard | `node_labels_constant_and_binary` | Every run must have one binary 16-router bitmap. |
| PASS | hard | `scenario_metadata_constant` | split/active_cores/profile/seed/strength/attackers must be constant within runs. |
| PASS | hard | `attacker_metadata_matches_node_labels` | Normal runs must have zero bitmap; attacks must match attackers metadata. |
| PASS | hard | `expected_run_counts_by_split` | observed={'train': 45, 'val': 13, 'test': 13} expected={'train': 45, 'val': 13, 'test': 13} |
| PASS | hard | `edge_index_shape` | observed=(2, 64) expected=(2, 64) |
| PASS | hard | `edge_index_integer_dtype` | dtype=int64 |
| PASS | hard | `edge_index_router_ids` | min=0 max=15 |
| PASS | hard | `edge_index_no_duplicates` | edges=64 unique=64 |
| PASS | hard | `edge_index_self_loops` | observed=16 expected=16 |
| PASS | hard | `edge_index_nonself_edges` | observed=48 expected=48 |
| PASS | hard | `edge_index_all_nodes_present` | nodes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] |
| PASS | hard | `edge_index_connected` | Connectivity checked after ignoring edge direction. |
| PASS | warning | `edge_index_reciprocal_nonself` | reciprocal_nonself=48 nonself=48 |
| PASS | hard | `feature_values_finite` | nonfinite_total=0 |
| PASS | hard | `feature_values_in_range` | out_of_range_features=[] |
| PASS | hard | `aggregate_directional_count_consistency` | total_violations=0; tolerance=1e-06 |
| PASS | hard | `hard_runs_present_and_correct` | Both dominant hard runs must exist with expected split and labels. |

## Documented limitations

- Directional IFD contains exact 1.0 values: 106351415/299267840 (35.537201%). Their physical causes are overloaded.
- Boundary-port diagnostics use a provisional row-major 4x4 compass mapping and are not treated as hard failures.
- Final arrays alone cannot distinguish invalid, idle, no-gap, and clipped-long-gap IFD states.
- The dataset has many overlapping windows but only 71 independent simulation runs.
- The current V3 test split has been used for diagnosis and is no longer an untouched final publication holdout.
- Archive-derived 1980 modification times cannot establish the original generation date; hashes should be used for identity.

## Interpretation

The final arrays match the recovered builder rules. The remaining findings are representation, coverage, or provenance limitations rather than evidence of array corruption.

## Output tables

- `tables/stage8d_run_structure.csv`
- `tables/stage8d_run_metadata_consistency.csv`
- `tables/stage8d_feature_range_summary.csv`
- `tables/stage8d_count_consistency.csv`
- `tables/stage8d_ifd_one_rate_by_feature.csv`
- `tables/stage8d_ifd_one_rate_by_router.csv`
- `tables/stage8d_ifd_one_rate_by_split_class.csv`
- `tables/stage8d_boundary_port_audit.csv`
- `tables/stage8d_hard_run_audit.csv`
- `tables/stage8d_summary.json`

## What this audit cannot prove

- The physical meaning of headerless raw-trace columns without the gem5 instrumentation source.
- Exact attack-worker startup timing without `traffic_worker_v2`.
- Whether each IFD value of 1.0 means invalid, idle, no-gap, or clipped-long-gap using only the final tensor.
- Physical compass-direction mapping without the graph and trace mapping sources.

## Summary snapshot

```json
{
  "checks_passed": 30,
  "checks_total": 30,
  "chunk_size": 512,
  "dataset_root": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3",
  "generated_at": "2026-07-18T18:28:36.795199+00:00",
  "graph": {
    "all_nodes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15
    ],
    "connected": true,
    "dtype": "int64",
    "edge_count": 64,
    "maximum_router_id": 15,
    "minimum_router_id": 0,
    "nonself_edges": 48,
    "reciprocal_nonself_edges": 48,
    "self_loops": 16,
    "shape": [
      2,
      64
    ],
    "unique_edge_count": 64
  },
  "hard_failures": [],
  "input_files": {
    "active_cores": {
      "dtype": "<U14",
      "path": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3/active_cores.npy",
      "shape": [
        233803
      ],
      "size_bytes": 13093096
    },
    "attackers": {
      "dtype": "<U7",
      "path": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3/attackers.npy",
      "shape": [
        233803
      ],
      "size_bytes": 6546612
    },
    "edge_index": {
      "dtype": "int64",
      "path": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3/edge_index.npy",
      "shape": [
        2,
        64
      ],
      "size_bytes": 1152
    },
    "end_epoch": {
      "dtype": "int32",
      "path": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3/end_epoch.npy",
      "shape": [
        233803
      ],
      "size_bytes": 935340
    },
    "feature_cols": {
      "dtype": "<U22",
      "path": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3/feature_cols.npy",
      "shape": [
        24
      ],
      "size_bytes": 2240
    },
    "profile": {
      "dtype": "<U7",
      "path": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3/profile.npy",
      "shape": [
        233803
      ],
      "size_bytes": 6546612
    },
    "run_id": {
      "dtype": "<U39",
      "path": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3/run_id.npy",
      "shape": [
        233803
      ],
      "size_bytes": 36473396
    },
    "seed": {
      "dtype": "<U2",
      "path": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3/seed.npy",
      "shape": [
        233803
      ],
      "size_bytes": 1870552
    },
    "split": {
      "dtype": "<U5",
      "path": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3/split.npy",
      "shape": [
        233803
      ],
      "size_bytes": 4676188
    },
    "strength": {
      "dtype": "<U2",
      "path": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3/strength.npy",
      "shape": [
        233803
      ],
      "size_bytes": 1870552
    },
    "x": {
      "dtype": "float32",
      "path": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3/x.npy",
      "shape": [
        233803,
        16,
        8,
        24
      ],
      "size_bytes": 2872971392
    },
    "y_graph": {
      "dtype": "float32",
      "path": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3/y_graph.npy",
      "shape": [
        233803
      ],
      "size_bytes": 935340
    },
    "y_node": {
      "dtype": "float32",
      "path": "/home/zira/tools/architecture/gem5/experiments/paper1_temporal_graphs_ports_v3/y_node.npy",
      "shape": [
        233803,
        16
      ],
      "size_bytes": 14963520
    }
  },
  "limitations": [
    "Directional IFD contains exact 1.0 values: 106351415/299267840 (35.537201%). Their physical causes are overloaded.",
    "Boundary-port diagnostics use a provisional row-major 4x4 compass mapping and are not treated as hard failures.",
    "Final arrays alone cannot distinguish invalid, idle, no-gap, and clipped-long-gap IFD states.",
    "The dataset has many overlapping windows but only 71 independent simulation runs.",
    "The current V3 test split has been used for diagnosis and is no longer an untouched final publication holdout.",
    "Archive-derived 1980 modification times cannot establish the original generation date; hashes should be used for identity."
  ],
  "numpy": "2.5.1",
  "output_root": "/home/zira/research/projects/GNN-2d/reports/v3_failure_analysis/stage8d_final_consistency",
  "platform": "Linux-6.8.0-134-generic-x86_64-with-glibc2.39",
  "python": "3.12.3",
  "run_segments": 71,
  "unique_runs": 71,
  "verdict": "PASS WITH DOCUMENTED LIMITATIONS",
  "warning_findings": [],
  "x_dtype": "float32",
  "x_memory_mapped": true,
  "x_shape": [
    233803,
    16,
    8,
    24
  ]
}
```
