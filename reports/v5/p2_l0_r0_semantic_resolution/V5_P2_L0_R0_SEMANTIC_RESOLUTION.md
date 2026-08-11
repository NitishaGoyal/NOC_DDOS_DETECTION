# V5 P2 L0-R0 Semantic Resolution

## Directly resolved facts

- Topology: **4×4 2D mesh, 16 routers**
- Router numbering: **row-major**
- Router mapping: **router_id = row × 4 + column**
- Directed physical edges: **48**
- Edge order: **lexicographic by (source, destination)**
- Count classes: **K1, K2, K3, K4**
- Frozen training loss weights: `{'attack': 1.0, 'count': 0.5, 'source': 1.0, 'transit': 0.5, 'victim': 0.75, 'path': 0.5}`

## Resolution status

- router_coordinate_mapping_resolved: **true**
- physical_mesh_and_edge_order_resolved: **true**
- count_class_semantics_resolved: **true**
- loss_weights_resolved: **true**
- routing_dimension_order_resolved: **true**
- source_victim_route_construction_resolved: **true**
- transit_endpoint_exclusion_resolved: **true**
- path_endpoint_inclusion_resolved: **true**

## Contract topology arrays

### V5_P2_G1A_R2A_CANONICAL_ADJACENCY_MATRIX.npy

```json
{
  "dtype": "uint8",
  "path": "/home/zira/research/projects/GNN-2d/reports/v5/p2_g1a_r2a_canonical_static_topology_contract/V5_P2_G1A_R2A_CANONICAL_ADJACENCY_MATRIX.npy",
  "sha256": "a0deaedb2d92398154be3e6a62f03615e5fc15dfc1b48590a031b940ad6eda84",
  "shape": [
    16,
    16
  ],
  "values": [
    [
      0,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      0
    ]
  ]
}
```

### V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy

```json
{
  "dtype": "int64",
  "path": "/home/zira/research/projects/GNN-2d/reports/v5/p2_g1a_r2a_canonical_static_topology_contract/V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
  "sha256": "f6b8050bc158de509b0ff1c5d1d7cb1ffe32c08b0f2287398270b1b891b57aff",
  "shape": [
    2,
    48
  ],
  "values": [
    [
      0,
      0,
      1,
      1,
      1,
      2,
      2,
      2,
      3,
      3,
      4,
      4,
      4,
      5,
      5,
      5,
      5,
      6,
      6,
      6,
      6,
      7,
      7,
      7,
      8,
      8,
      8,
      9,
      9,
      9,
      9,
      10,
      10,
      10,
      10,
      11,
      11,
      11,
      12,
      12,
      13,
      13,
      13,
      14,
      14,
      14,
      15,
      15
    ],
    [
      1,
      4,
      0,
      2,
      5,
      1,
      3,
      6,
      2,
      7,
      0,
      5,
      8,
      1,
      4,
      6,
      9,
      2,
      5,
      7,
      10,
      3,
      6,
      11,
      4,
      9,
      12,
      5,
      8,
      10,
      13,
      6,
      9,
      11,
      14,
      7,
      10,
      15,
      8,
      13,
      9,
      12,
      14,
      10,
      13,
      15,
      11,
      14
    ]
  ]
}
```

### V5_P2_G1A_R2A_ROUTER_COORDINATES_ROW_MAJOR.npy

```json
{
  "dtype": "int64",
  "path": "/home/zira/research/projects/GNN-2d/reports/v5/p2_g1a_r2a_canonical_static_topology_contract/V5_P2_G1A_R2A_ROUTER_COORDINATES_ROW_MAJOR.npy",
  "sha256": "b0cc6b9e894f2b076f02ae2cf15541f2786c37930ed1facccb2574ea4feec2e1",
  "shape": [
    16,
    2
  ],
  "values": [
    [
      0,
      0
    ],
    [
      0,
      1
    ],
    [
      0,
      2
    ],
    [
      0,
      3
    ],
    [
      1,
      0
    ],
    [
      1,
      1
    ],
    [
      1,
      2
    ],
    [
      1,
      3
    ],
    [
      2,
      0
    ],
    [
      2,
      1
    ],
    [
      2,
      2
    ],
    [
      2,
      3
    ],
    [
      3,
      0
    ],
    [
      3,
      1
    ],
    [
      3,
      2
    ],
    [
      3,
      3
    ]
  ]
}
```

## Source evidence: Routing Order

### `scripts/failure_analysis/export_chrono_b1_validation_predictions.py:215` (match at line 218)

```python
        "nodes": int(node_prob.shape[1]),
        "graph_prob_shape": list(graph_prob.shape),
        "node_prob_shape": list(node_prob.shape),
        "real_index_first": int(real_index[0]),
        "real_index_last": int(real_index[-1]),
        "graph_prob_min": float(graph_prob.min()),
        "graph_prob_max": float(graph_prob.max()),
```

### `scripts/failure_analysis/v4/audit_v4_a0_2_manifest_array_alignment.py:578` (match at line 581)

```python

    first_indices = np.arange(args.expected_runs, dtype=np.int64) * args.expected_samples_per_run

    run_index_first = np.asarray(run_index[first_indices], dtype=np.int64)
    expected_run_index = np.arange(args.expected_runs, dtype=np.int64)
    run_index_order_ok = np.array_equal(run_index_first, expected_run_index)

```

### `scripts/failure_analysis/v4/audit_v4_a0_2_manifest_array_alignment.py:580` (match at line 583)

```python

    run_index_first = np.asarray(run_index[first_indices], dtype=np.int64)
    expected_run_index = np.arange(args.expected_runs, dtype=np.int64)
    run_index_order_ok = np.array_equal(run_index_first, expected_run_index)

    completed_count_ok = len(completed_runs) == args.expected_runs
    completed_unique_ok = len(set(completed_runs)) == len(completed_runs)
```

### `scripts/failure_analysis/v4/audit_v4_a0_2_manifest_array_alignment.py:833` (match at line 836)

```python
            len(completed_missing_from_manifest) == 0
            and len(manifest_extra_vs_completed) == 0
        ),
        "run_index_first_samples_are_0_through_907": run_index_order_ok,
        "metadata_run_order_matches_completed_when_available": (
            metadata_order_matches_completed is not False
        ),
```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:407` (match at line 410)

```python

    selected: dict[str, Any] | None
    if feasible:
        # Shortest valid operational latency first, then strongest safety and
        # localization performance.
        selected = min(
            feasible,
```

### `scripts/v5/p2/audit_v5_p2_a1_tensor_schema_nontest.py:758` (match at line 761)

```python
        "sample_policy": {
            "splits": ["train", "validation"],
            "pair_positions_per_split": [
                "lexicographically_first",
                "lexicographically_middle",
                "lexicographically_last",
            ],
```

### `scripts/v5/p2/freeze_v5_p2_c2a_legal_noc_decoder_amendment.py:151` (match at line 154)

```python
                "data_access": "none",
            },
            {
                "stage": "V5_P2_L1_XY_ROUTE_LIBRARY_AND_UNIT_TESTS",
                "purpose": "generate and verify all 240 directed 4x4 XY routes",
                "data_access": "none",
            },
```

### `scripts/v5/p2/freeze_v5_p2_c2a_legal_noc_decoder_amendment.py:152` (match at line 155)

```python
            },
            {
                "stage": "V5_P2_L1_XY_ROUTE_LIBRARY_AND_UNIT_TESTS",
                "purpose": "generate and verify all 240 directed 4x4 XY routes",
                "data_access": "none",
            },
            {
```

### `scripts/v5/p2/freeze_v5_p2_c2a_legal_noc_decoder_amendment.py:210` (match at line 213)

```python
        "decoder_scope": {
            "position": "after B3 causal Conv1D-count4 heads",
            "topology": "4x4 2D mesh",
            "routing": "deterministic XY",
            "attacker_count": [1, 2, 3, 4],
            "one_victim_assignment_per_source": True,
            "shared_victims_allowed": True,
```

### `scripts/v5/p2/freeze_v5_p2_c2a_legal_noc_decoder_amendment.py:237` (match at line 240)

```python
                "input logit semantics",
                "source/transit/victim/path union semantics",
                "route-pair hypothesis semantics",
                "XY route legality",
                "multi-attacker overlap semantics",
                "count handling",
                "structured scoring equation",
```

### `scripts/v5/p2/freeze_v5_p2_c2a_legal_noc_decoder_amendment.py:296` (match at line 299)

```python

## Initial scope

- 4x4 2D mesh with deterministic XY routing.
- K1-K4 sources.
- One victim assignment per source.
- Shared victims and overlapping legal routes are allowed.
```

### `scripts/v5/p2/freeze_v5_p2_g0_graph_baseline_rtl_handoff_protocol.py:968` (match at line 971)

```python
        "",
        "`G1 source-only -> G2 graph detection -> G3 role-aware/full "
        "multitask -> G4 five-seed selection -> G5 RTL selection contract -> "
        "G6 RTL handoff -> return to L0 Legal XY decoder contract`",
        "",
        "## Mandatory operators",
        "",
```

## Source evidence: Route Construction

### `src/models/v5_frozen_b3_conv1d_only.py:63` (match at line 66)

```python
        [B]
    count_logits:
        [B,3]
    source_logits, transit_logits, victim_logits, path_logits:
        each [B,16]
    """

```

### `src/models/v5_p2_b3_conv1d_only_count4.py:73` (match at line 76)

```python
        [B]
    count_logits:
        [B,4], classes map to raw active counts [1,2,3,4]
    source_logits, transit_logits, victim_logits, path_logits:
        each [B,16]
    """

```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:492` (match at line 495)

```python
    return mapping


def router_groups_for_run(
    metadata: dict[str, Any],
    neighbor_map: dict[int, list[int]],
    router_count: int,
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:580` (match at line 583)

```python
# Selected-run router summaries
# ---------------------------------------------------------------------------

def summarize_router_features(
    run_data: np.ndarray,
    metadata: dict[str, Any],
    taxonomy: pd.DataFrame,
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:918` (match at line 921)

```python
# Router group summaries
# ---------------------------------------------------------------------------

def summarize_router_groups(
    run_data: np.ndarray,
    metadata: dict[str, Any],
    taxonomy: pd.DataFrame,
```

### `scripts/failure_analysis/audit_v3_stage8d_final_consistency.py:239` (match at line 242)

```python
    return bool(np.all(array[start:stop] == first)), first


def bitmap_routers(bitmap: np.ndarray, tolerance: float) -> list[int]:
    return [int(i) for i in np.flatnonzero(np.asarray(bitmap) > 0.5)]


```

### `scripts/failure_analysis/audit_v3_stage8d_final_consistency.py:272` (match at line 275)

```python
    return digest.hexdigest()


def expected_invalid_direction(router: int, direction: str) -> bool:
    row, col = divmod(router, 4)
    return {
        "local": False,
```

### `scripts/failure_analysis/v4/analyze_v4_a1_failures.py:172` (match at line 175)

```python
    return "-".join(str(int(v)) for v in values)


def router_coord(router: int) -> tuple[int, int]:
    return int(router) // MESH_SIDE, int(router) % MESH_SIDE


```

### `scripts/failure_analysis/v4/analyze_v4_a1_failures.py:182` (match at line 185)

```python
    return abs(ar - br) + abs(ac - bc)


def router_topology(router: int) -> str:
    row, col = router_coord(router)
    borders = int(row in (0, MESH_SIDE - 1)) + int(col in (0, MESH_SIDE - 1))
    if borders == 2:
```

### `scripts/failure_analysis/v4/analyze_v4_a1_failures.py:192` (match at line 195)

```python
    return "interior"


def direction_valid(router: int, direction: str) -> bool:
    row, col = router_coord(router)
    return {
        "north": row > 0,
```

### `scripts/failure_analysis/v4/analyze_v4_a1_failures.py:202` (match at line 205)

```python
    }[direction]


def on_any_shortest_path(router: int, sources: Sequence[int], victims: Sequence[int]) -> bool:
    for source in sources:
        for victim in victims:
            if manhattan(source, router) + manhattan(router, victim) == manhattan(source, victim):
```

### `scripts/failure_analysis/v4/analyze_v4_a1_failures.py:205` (match at line 208)

```python
def on_any_shortest_path(router: int, sources: Sequence[int], victims: Sequence[int]) -> bool:
    for source in sources:
        for victim in victims:
            if manhattan(source, router) + manhattan(router, victim) == manhattan(source, victim):
                return True
    return False

```

### `scripts/failure_analysis/v4/analyze_v4_a1_failures.py:770` (match at line 773)

```python
    return result


def router_exposure_analysis(
    records: Sequence[RunRecord],
    run_slices: Mapping[int, tuple[int, int]],
    out: Path,
```

### `scripts/failure_analysis/v4/analyze_v4_a1_remaining_diagnostics.py:110` (match at line 113)

```python
    return -np.sum(q * np.log(q) + (1.0 - q) * np.log(1.0 - q), axis=axis)


def topology_of_router(router: int) -> str:
    if router in {0, 3, 12, 15}:
        return "corner"
    if router in {5, 6, 9, 10}:
```

### `scripts/failure_analysis/v4/run_v4_a3_complete_analysis.py:292` (match at line 295)

```python
    return rows


def per_router_metrics(y_node: np.ndarray, node_pred: np.ndarray) -> list[dict[str, Any]]:
    truth = y_node >= 0.5
    pred = node_pred.astype(bool)
    rows = []
```

### `scripts/failure_analysis/v4/run_v4_pretraining_diagnostic.py:517` (match at line 520)

```python
    return summary, {"matched_control_audit": audit, "matched_control_failures": failures, "active_core_shortcut_cases": shortcuts}


def router_direction_valid(router, direction, mesh_cols=4):
    row, col = divmod(router, mesh_cols)
    return {"north": row > 0, "east": col < mesh_cols - 1, "south": row < mesh_cols - 1, "west": col > 0}[direction]

```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:141` (match at line 144)

```python
    return out


def router_three_of_four(
    pred: np.ndarray,
    run_index: np.ndarray,
) -> np.ndarray:
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:138` (match at line 141)

```python
    return out


def router_three_of_four(
    pred: np.ndarray,
    run_index: np.ndarray,
) -> np.ndarray:
```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:116` (match at line 119)

```python
    return out


def router_three_of_four(
    pred: np.ndarray,
    run_index: np.ndarray,
) -> np.ndarray:
```

### `scripts/failure_analysis/v4_a3_13_branch_attribution.py:252` (match at line 255)

```python
    return out


def router_three_of_four(
    pred: np.ndarray,
    run_index: np.ndarray,
) -> np.ndarray:
```

### `scripts/failure_analysis/v4_a3_15_directional_source_transit.py:201` (match at line 204)

```python
    return out


def router_three_of_four(
    pred: np.ndarray,
    run_index: np.ndarray,
) -> np.ndarray:
```

### `scripts/failure_analysis/v4_a3_15_directional_source_transit.py:479` (match at line 482)

```python
    return matrix.astype(np.float32), metric_names


def role_for_router(
    run_meta: dict[str, Any],
    router: int,
) -> str:
```

### `scripts/v4/a4a/v4_a4a_2b_validation_selector.py:276` (match at line 279)

```python
    return None


def maximum_router_streak(predicted_masks: np.ndarray) -> int:
    maximum = 0
    for router in range(NUM_ROUTERS):
        active = (predicted_masks & (1 << router)) != 0
```

### `scripts/v4/a4a/v4_a4a_2b_validation_selector.py:285` (match at line 288)

```python
    return int(maximum)


def has_persistent_router(
    predicted_masks: np.ndarray,
    streak_required: int,
) -> bool:
```

### `scripts/v4/a4a/v4_a4a_2b_validation_selector_v2.py:276` (match at line 279)

```python
    return None


def maximum_router_streak(predicted_masks: np.ndarray) -> int:
    maximum = 0
    for router in range(NUM_ROUTERS):
        active = (predicted_masks & (1 << router)) != 0
```

### `scripts/v4/a4a/v4_a4a_2b_validation_selector_v2.py:285` (match at line 288)

```python
    return int(maximum)


def has_persistent_router(
    predicted_masks: np.ndarray,
    streak_required: int,
) -> bool:
```

### `scripts/v4/a4a/v4_a4a_final_closure.py:429` (match at line 432)

```python
                "were insufficient."
            ),
            "dataset_implication": (
                "V4 does not provide enough explicit source/transit/victim "
                "and route-transition evidence for reliable attacker-set "
                "cardinality and role separation."
            ),
```

### `scripts/v4/a4a/v4_a4a_final_closure.py:537` (match at line 540)

```python

A4a retained a useful precision/safety operating point, but it did not recover
the A3.12 cardinality gap. The remaining failure is consistent with missing
source/transit/victim and route-transition semantics in V4 rather than a small
checkpoint-selection error.

## Final boundary
```

### `scripts/v4/common/v4_a3_sourcepreserve.py:348` (match at line 351)

```python
    return checks


def build_normalized_adjacency(edge_index: np.ndarray, num_nodes: int = NUM_ROUTERS) -> torch.Tensor:
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for source, destination in zip(edge_index[0], edge_index[1]):
        adjacency[int(destination), int(source)] = 1.0
```

### `scripts/v5/baselines/train_v5_p0_b4_gcn_only.py:283` (match at line 286)

```python
    }
    for role, key in ROLE_KEYS.items():
        losses[role] = F.binary_cross_entropy_with_logits(outputs[role], batch[key].to(device=device, dtype=torch.float32), pos_weight=pos_weights[role])
    total = losses["attack"] + 0.5 * losses["count"] + losses["source"] + 0.5 * losses["transit"] + 0.75 * losses["victim"] + 0.5 * losses["path"]
    return total, {k: float(v.detach()) for k, v in losses.items()}


```

### `scripts/v5/baselines/train_v5_p0_b4_gcn_only.py:682` (match at line 685)

```python

    d = comparison["validation_b4_minus_b2"]
    print("\n===== B4 FINAL RESULTS =====")
    print("train:", f"g_bal_acc={train_metrics['graph']['balanced_accuracy']:.4f}", f"g_f1={train_metrics['graph']['f1']:.4f}", f"g_fpr={train_metrics['graph']['fpr']:.4f}", f"count_macro_f1={train_metrics['count']['macro_f1']:.4f}", f"source_f1_attack={train_metrics['roles']['source']['attack_windows']['node_f1']:.4f}", f"victim_f1_attack={train_metrics['roles']['victim']['attack_windows']['node_f1']:.4f}", f"all_exact={train_metrics['all_tasks_exact']:.4f}")
    print("validation:", f"g_bal_acc={val_metrics['graph']['balanced_accuracy']:.4f}", f"g_f1={val_metrics['graph']['f1']:.4f}", f"g_fpr={val_metrics['graph']['fpr']:.4f}", f"count_macro_f1={val_metrics['count']['macro_f1']:.4f}", f"source_f1_attack={val_metrics['roles']['source']['attack_windows']['node_f1']:.4f}", f"victim_f1_attack={val_metrics['roles']['victim']['attack_windows']['node_f1']:.4f}", f"all_exact={val_metrics['all_tasks_exact']:.4f}")
    print("validation_delta_B4_minus_B2:", f"g_bal_acc={d['graph_balanced_accuracy']:+.4f}", f"g_f1={d['graph_f1']:+.4f}", f"g_fpr={d['graph_fpr']:+.4f}", f"count_macro_f1={d['count_macro_f1']:+.4f}", f"source_f1={d['source_node_f1_attack']:+.4f}", f"transit_f1={d['transit_node_f1_attack']:+.4f}", f"victim_f1={d['victim_node_f1_attack']:+.4f}", f"path_f1={d['path_node_f1_attack']:+.4f}", f"all_exact={d['all_tasks_exact']:+.4f}")
    print("failure_count: 0")
```

### `scripts/v5/baselines/train_v5_p0_b4_gcn_only.py:683` (match at line 686)

```python
    d = comparison["validation_b4_minus_b2"]
    print("\n===== B4 FINAL RESULTS =====")
    print("train:", f"g_bal_acc={train_metrics['graph']['balanced_accuracy']:.4f}", f"g_f1={train_metrics['graph']['f1']:.4f}", f"g_fpr={train_metrics['graph']['fpr']:.4f}", f"count_macro_f1={train_metrics['count']['macro_f1']:.4f}", f"source_f1_attack={train_metrics['roles']['source']['attack_windows']['node_f1']:.4f}", f"victim_f1_attack={train_metrics['roles']['victim']['attack_windows']['node_f1']:.4f}", f"all_exact={train_metrics['all_tasks_exact']:.4f}")
    print("validation:", f"g_bal_acc={val_metrics['graph']['balanced_accuracy']:.4f}", f"g_f1={val_metrics['graph']['f1']:.4f}", f"g_fpr={val_metrics['graph']['fpr']:.4f}", f"count_macro_f1={val_metrics['count']['macro_f1']:.4f}", f"source_f1_attack={val_metrics['roles']['source']['attack_windows']['node_f1']:.4f}", f"victim_f1_attack={val_metrics['roles']['victim']['attack_windows']['node_f1']:.4f}", f"all_exact={val_metrics['all_tasks_exact']:.4f}")
    print("validation_delta_B4_minus_B2:", f"g_bal_acc={d['graph_balanced_accuracy']:+.4f}", f"g_f1={d['graph_f1']:+.4f}", f"g_fpr={d['graph_fpr']:+.4f}", f"count_macro_f1={d['count_macro_f1']:+.4f}", f"source_f1={d['source_node_f1_attack']:+.4f}", f"transit_f1={d['transit_node_f1_attack']:+.4f}", f"victim_f1={d['victim_node_f1_attack']:+.4f}", f"path_f1={d['path_node_f1_attack']:+.4f}", f"all_exact={d['all_tasks_exact']:+.4f}")
    print("failure_count: 0")
    print("warning_count: 0")
```

### `scripts/v5/baselines/train_v5_p0_b4_gcn_only.py:684` (match at line 687)

```python
    print("\n===== B4 FINAL RESULTS =====")
    print("train:", f"g_bal_acc={train_metrics['graph']['balanced_accuracy']:.4f}", f"g_f1={train_metrics['graph']['f1']:.4f}", f"g_fpr={train_metrics['graph']['fpr']:.4f}", f"count_macro_f1={train_metrics['count']['macro_f1']:.4f}", f"source_f1_attack={train_metrics['roles']['source']['attack_windows']['node_f1']:.4f}", f"victim_f1_attack={train_metrics['roles']['victim']['attack_windows']['node_f1']:.4f}", f"all_exact={train_metrics['all_tasks_exact']:.4f}")
    print("validation:", f"g_bal_acc={val_metrics['graph']['balanced_accuracy']:.4f}", f"g_f1={val_metrics['graph']['f1']:.4f}", f"g_fpr={val_metrics['graph']['fpr']:.4f}", f"count_macro_f1={val_metrics['count']['macro_f1']:.4f}", f"source_f1_attack={val_metrics['roles']['source']['attack_windows']['node_f1']:.4f}", f"victim_f1_attack={val_metrics['roles']['victim']['attack_windows']['node_f1']:.4f}", f"all_exact={val_metrics['all_tasks_exact']:.4f}")
    print("validation_delta_B4_minus_B2:", f"g_bal_acc={d['graph_balanced_accuracy']:+.4f}", f"g_f1={d['graph_f1']:+.4f}", f"g_fpr={d['graph_fpr']:+.4f}", f"count_macro_f1={d['count_macro_f1']:+.4f}", f"source_f1={d['source_node_f1_attack']:+.4f}", f"transit_f1={d['transit_node_f1_attack']:+.4f}", f"victim_f1={d['victim_node_f1_attack']:+.4f}", f"path_f1={d['path_node_f1_attack']:+.4f}", f"all_exact={d['all_tasks_exact']:+.4f}")
    print("failure_count: 0")
    print("warning_count: 0")
    print("test_split_accessed: false")
```

### `scripts/v5/baselines/train_v5_p0_b6_dual_temporal_fusion_no_gcn.py:33` (match at line 36)

```python
    -> ReLU

Outputs:
    shared node heads for source, transit, victim, and path;
    mean+max router pooling;
    graph heads for attack and attacker count.

```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic.py:6` (match at line 9)

```python
-------------------
Can a NoC-specific, directional, gated, residual one-hop graph operation add
useful network context after B3's causal Conv1D encoder without smoothing away
router-local source and victim evidence?

Controlled comparison against B3
--------------------------------
```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic.py:19` (match at line 22)

```python
    exact same B3 temporal encoder and router-local node representation
    -> preserve h_local unchanged
    -> add one-hop directional gated residual context h_ctx
    -> source/victim heads use h_local only
    -> transit/path heads use concat(h_local, h_ctx)
    -> graph/count heads pool h_ctx

```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic.py:1291` (match at line 1294)

```python
        "requirements": requirements,
        "promotion_rule": (
            "All hard requirements must pass. A higher all-task exact score "
            "cannot compensate for detection, source, or victim failure."
        ),
    }

```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic.py:1715` (match at line 1718)

```python
                .cpu()
                .item()
            ),
            "source_victim_use_local_only": True,
            "transit_path_use_local_and_context": True,
            "graph_count_use_context": True,
            "standard_gcn_normalization_used": False,
```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic.py:1808` (match at line 1811)

```python
        "status": "COMPLETE",
        "scientific_question": (
            "Can directional, port-aware, gated, residual one-hop graph "
            "context improve B3 without smoothing away source/victim evidence?"
        ),
        "candidate_decision": candidate_decision,
        "success_gate": success_gate,
```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic.py:1879` (match at line 1882)

```python
            "same_binary_threshold": True,
            "same_temporal_encoder": True,
            "same_node_post_projection": True,
            "source_victim_local_branch_preserved": True,
            "no_temporal_mean_branch": True,
            "parameter_count_not_matched": True,
            "predeclared_architectural_change": (
```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic.py:1884` (match at line 1887)

```python
            "parameter_count_not_matched": True,
            "predeclared_architectural_change": (
                "add one directional gated residual context module; preserve "
                "B3 local evidence for source and victim heads"
            ),
        },
        "train_target_statistics": train_stats,
```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic_b7b.py:6` (match at line 9)

```python
-------------------
Can a NoC-specific, directional, gated, residual one-hop graph operation add
useful network context after B3's causal Conv1D encoder without smoothing away
router-local source and victim evidence?

Controlled comparison against B3
--------------------------------
```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic_b7b.py:19` (match at line 22)

```python
    exact same B3 temporal encoder and router-local node representation
    -> preserve h_local unchanged
    -> add one-hop directional gated residual context h_ctx
    -> source/victim heads use h_local only
    -> transit/path heads use concat(h_local, h_ctx)
    -> graph/count heads pool h_ctx

```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic_b7b.py:1294` (match at line 1297)

```python
        "requirements": requirements,
        "promotion_rule": (
            "All hard requirements must pass. A higher all-task exact score "
            "cannot compensate for detection, source, or victim failure."
        ),
    }

```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic_b7b.py:1718` (match at line 1721)

```python
                .cpu()
                .item()
            ),
            "source_victim_use_local_only": True,
            "transit_path_use_local_and_context": True,
            "graph_count_use_context": True,
            "standard_gcn_normalization_used": False,
```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic_b7b.py:1811` (match at line 1814)

```python
        "status": "COMPLETE",
        "scientific_question": (
            "Can directional, port-aware, gated, residual one-hop graph "
            "context improve B3 without smoothing away source/victim evidence?"
        ),
        "candidate_decision": candidate_decision,
        "success_gate": success_gate,
```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic_b7b.py:1882` (match at line 1885)

```python
            "same_binary_threshold": True,
            "same_temporal_encoder": True,
            "same_node_post_projection": True,
            "source_victim_local_branch_preserved": True,
            "no_temporal_mean_branch": True,
            "parameter_count_not_matched": True,
            "predeclared_architectural_change": (
```

### `scripts/v5/baselines/train_v5_p0_b7a_directional_gated_graph_diagnostic_b7b.py:1887` (match at line 1890)

```python
            "parameter_count_not_matched": True,
            "predeclared_architectural_change": (
                "add one directional gated residual context module; preserve "
                "B3 local evidence for source and victim heads"
            ),
        },
        "train_target_statistics": train_stats,
```

### `scripts/v5/common/audit_v5_p0_tensor_handover.py:51` (match at line 54)

```python
    "snapshot_tick", "tick", "epoch", "epoch_id", "router", "router_id",
    "case", "case_id", "pair", "pair_id", "run", "run_id", "seed",
    "mode", "split", "attack_active", "attacker_count",
    "active_attack_count", "source_router", "victim_router", "packet_id",
    "flit_id", "address", "candidate_getx_count", "role_mask", "y_attack",
    "y_attacker_count", "y_source", "y_transit", "y_victim",
    "y_attack_path",
```

### `scripts/v5/common/audit_v5_p0_tensor_handover.py:53` (match at line 56)

```python
    "mode", "split", "attack_active", "attacker_count",
    "active_attack_count", "source_router", "victim_router", "packet_id",
    "flit_id", "address", "candidate_getx_count", "role_mask", "y_attack",
    "y_attacker_count", "y_source", "y_transit", "y_victim",
    "y_attack_path",
}

```

### `scripts/v5/common/audit_v5_p0_tensor_handover.py:61` (match at line 64)

```python
    "pair_id", "run_id", "case_id", "packet_id", "flit_id",
    "candidate_getx", "active_attack_count", "attack_active", "y_source",
    "y_transit", "y_victim", "y_attack_path", "role_mask",
    "victim_router", "source_router",
]


```

### `scripts/v5/common/audit_v5_p0_tensor_handover.py:322` (match at line 325)

```python
    if epoch_id.numel() > 1 and not bool(torch.all(epoch_id[1:] > epoch_id[:-1])):
        failures.append(f"{path.name}: epoch_id is not strictly increasing")

    for key in ("y_attack", "y_source", "y_transit", "y_victim", "y_attack_path"):
        check_binary(tensors[key], f"{path.name}:{key}", failures)

    y_attack = tensors["y_attack"].bool()
```

### `scripts/v5/common/audit_v5_p0_tensor_handover.py:342` (match at line 345)

```python
                f"{int(y_count.max().item())}"
            )

    expected_path = y_source | y_transit | y_victim
    if not torch.equal(y_path, expected_path):
        mismatch = int((y_path != expected_path).sum().item())
        failures.append(
```

### `scripts/v5/common/audit_v5_p0_tensor_handover.py:402` (match at line 405)

```python
                    f"{path.name}: control run contains nonzero {key}"
                )

    source_victim = y_source & y_victim
    invalid_overlap = source_victim & (y_transit | (role_mask != 5))
    if bool(invalid_overlap.any()):
        failures.append(
```

### `scripts/v5/common/audit_v5_p0_tensor_handover.py:403` (match at line 406)

```python
                )

    source_victim = y_source & y_victim
    invalid_overlap = source_victim & (y_transit | (role_mask != 5))
    if bool(invalid_overlap.any()):
        failures.append(
            f"{path.name}: source=victim overlap violates role_mask=5 semantics"
```

### `scripts/v5/common/audit_v5_p0_tensor_handover.py:406` (match at line 409)

```python
    invalid_overlap = source_victim & (y_transit | (role_mask != 5))
    if bool(invalid_overlap.any()):
        failures.append(
            f"{path.name}: source=victim overlap violates role_mask=5 semantics"
        )

    row.update({
```

### `scripts/v5/common/audit_v5_p0_tensor_handover.py:424` (match at line 427)

```python
        "maximum_attacker_count": (
            int(y_count.max().item()) if y_count.numel() else 0
        ),
        "source_victim_overlap_entries": int(source_victim.sum().item()),
        "x_min": float(x.min().item()),
        "x_max": float(x.max().item()),
        "x_mean": float(x.mean().item()),
```

### `scripts/v5/common/audit_v5_p0_tensor_handover.py:893` (match at line 896)

```python
        })

    local_overlap_run_count = sum(
        int(row.get("source_victim_overlap_entries", 0)) > 0
        for row in run_rows
    )
    dual_attacker_run_count = sum(
```

### `scripts/v5/common/audit_v5_p0_tensor_handover.py:903` (match at line 906)

```python

    if local_overlap_run_count == 0:
        failures.append(
            "no source=victim overlap run found"
        )
    if dual_attacker_run_count == 0:
        failures.append(
```

### `scripts/v5/common/freeze_v5_p0_b0_r1_identifier_quarantine.py:9` (match at line 12)

```python
1. B0 found only the identifier/serialization HOLD.
2. The direct mode baseline and learned identifier probe have the same graph
   confusion matrix.
3. run_id and pair_id directly encode source, victim, and attacker count.
4. Every matched attack/control pair has equal run length.
5. Train and validation runs/pairs are disjoint.
6. A2/A3 permit only PRIMARY58, edge_index, and the audited physical mask as
```

### `scripts/v5/common/freeze_v5_p0_b0_r1_identifier_quarantine.py:326` (match at line 329)

```python
                - len(malformed_pairs)
                - len(unequal_length_pairs)
            ),
            "run_id_direct_source_victim_count_exact": True,
            "pair_id_direct_source_victim_count_exact": True,
        }

```

### `scripts/v5/common/freeze_v5_p0_b0_r1_identifier_quarantine.py:327` (match at line 330)

```python
                - len(unequal_length_pairs)
            ),
            "run_id_direct_source_victim_count_exact": True,
            "pair_id_direct_source_victim_count_exact": True,
        }

    pair_overlap = sorted(train_pairs & validation_pairs)
```

### `scripts/v5/common/freeze_v5_p0_b0_r1_identifier_quarantine.py:345` (match at line 348)

```python
        "classification": classification,
        "finding": (
            "The B0 HOLD is explained by provenance fields that encode run "
            "mode, scenario identity, source routers, victim routers, seed, "
            "and attacker count. These fields are not traffic measurements."
        ),
        "forbidden_learned_inputs": [
```

### `scripts/v5/common/freeze_v5_p0_b0_r1_identifier_quarantine.py:445` (match at line 448)

```python
`{classification}`

The original B0 HOLD remains preserved. It is classified as a provenance-only
shortcut because `run_id` and `pair_id` directly encode source, victim, and
attacker count, while `run_id`, file naming, and `mode` encode attack/control
run identity.

```

### `scripts/v5/common/freeze_v5_p0_b0_r1c_final_quarantine_release.py:208` (match at line 211)

```python
            or reduced["victim_exact"] >= thresholds["victim_exact"]
        ):
            failures.append(
                f"{name} source/victim exact localization violates release threshold"
            )

    # Frozen model input boundary.
```

### `scripts/v5/common/freeze_v5_p0_b0_r1c_final_quarantine_release.py:248` (match at line 251)

```python
            "file stem",
            "seed",
            "scenario identifier",
            "source/victim tokens parsed from identifiers",
            "dataset index",
            "run/file serialization order",
            "run length",
```

### `scripts/v5/common/run_v5_p0_b0_r1b_numeric_provenance_audit.py:859` (match at line 862)

```python
            or metric["roles"]["victim"]["exact_set"] >= 0.50
        ):
            hold_reasons.append(
                f"{name} source/victim exact localization too high"
            )

    serialization_metric = results["serialization_only"]
```

### `scripts/v5/common/run_v5_p0_b0_r1b_numeric_provenance_audit.py:920` (match at line 923)

```python
        "release_thresholds": {
            "critical_graph_balanced_accuracy_max_exclusive": 0.70,
            "critical_count_macro_f1_max_exclusive": 0.55,
            "critical_source_or_victim_exact_max_exclusive": 0.50,
        },
        "serialization_policy": (
            "serialization remains forbidden regardless of probe result"
```

### `scripts/v5/common/run_v5_p0_b0_shortcut_audit_suite.py:29` (match at line 32)

```python
import torch.nn.functional as F
from torch.utils.data import Dataset

ROLES = ("source", "transit", "victim", "path")
ROLE_KEYS = {
    "source": "y_source",
    "transit": "y_transit",
```

### `scripts/v5/common/run_v5_p0_b0_shortcut_audit_suite.py:361` (match at line 364)

```python
        outputs["attack"], truth["attack"].float(), pos_weight=weights["attack"]
    )
    loss = loss + 0.5 * F.cross_entropy(outputs["count"], truth["count"].long())
    coefficients = {"source": 1.0, "transit": 0.5, "victim": 0.75, "path": 0.5}
    for role in ROLES:
        loss = loss + coefficients[role] * F.binary_cross_entropy_with_logits(
            outputs[role], truth["roles"][role].float(), pos_weight=weights[role]
```

### `scripts/v5/common/run_v5_p0_b0_shortcut_audit_suite.py:732` (match at line 735)

```python
    }


def router_prior_baseline(
    train: dict[str, Any],
    validation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
```

### `scripts/v5/common/run_v5_p0_b0_shortcut_audit_suite.py:1262` (match at line 1265)

```python
        or router["roles"]["victim"]["attack_windows"]["node_f1"] >= 0.60
        or router["roles"]["victim"]["attack_windows"]["exact_set"] >= 0.50
    ):
        holds.append("static router prior is too predictive of source/victim")

    if (
        epoch["graph"]["balanced_accuracy"] >= 0.85
```

### `scripts/v5/final/evaluate_v5_p0_c3_one_shot_test.py:36` (match at line 39)

```python
STAGE = "V5_P0_C3_ONE_SHOT_LOCKED_TEST_EVALUATION"
COMPLETE = f"{STAGE}_COMPLETE"
ACCESS_STARTED = f"{STAGE}_TEST_ACCESS_STARTED"
ROLES = ("source", "transit", "victim", "path")
ROLE_KEYS = {
    "source": "y_source",
    "transit": "y_transit",
```

### `scripts/v5/final/freeze_v5_p0_c0_final_b3_protocol.py:331` (match at line 334)

```python
                    "closest threshold to 0.5",
                ],
            },
            "source_and_victim": {
                "calibration_scope": "ground-truth validation attack windows",
                "maximize_in_order": [
                    "node F1",
```

### `scripts/v5/final/freeze_v5_p0_c2_checkpoint_thresholds.py:38` (match at line 41)

```python

STAGE = "V5_P0_C2_FINAL_CHECKPOINT_THRESHOLD_FREEZE"
COMPLETE = f"{STAGE}_COMPLETE"
ROLES = ("source", "transit", "victim", "path")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
```

### `scripts/v5/final/freeze_v5_p0_c2_checkpoint_thresholds.py:404` (match at line 407)

```python
    role: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if role in ("source", "victim"):
        return max(
            rows,
            key=lambda row: (
```

### `scripts/v5/final/freeze_v5_p0_c2_checkpoint_thresholds.py:962` (match at line 965)

```python
                            "node_precision",
                            "closest_to_0.5",
                        ]
                        if role in ("source", "victim")
                        else [
                            "node_f1",
                            "exact_set",
```

### `scripts/v5/freeze/freeze_v5_p0_b8_b3_architecture.py:341` (match at line 344)

```python
            "projection": "Linear(74,64)->ReLU",
        },
        "node_heads": {
            "tasks": ["source", "transit", "victim", "attack_path"],
            "pattern": "Linear(64,32)->ReLU->Linear(32,1)",
        },
        "graph_heads": {
```

### `scripts/v5/freeze/freeze_v5_p0_b8_b3_architecture.py:362` (match at line 365)

```python
        "baseline_threshold": 0.5,
        "forbidden_inputs": [
            "run_id", "pair_id", "filename", "mode", "seed",
            "source/victim metadata", "scenario metadata",
            "serialization order", "run length", "window position",
            "router IDs", "router embeddings", "router coordinates",
            "status_flags", "stored standardized mask channels",
```

### `scripts/v5/p1/audit_v5_p1_a0_independent_dataset.py:62` (match at line 65)

```python
}

SUSPICIOUS_KEY_PATTERN = re.compile(
    r"(run|pair|file|name|mode|seed|source_id|victim_id|scenario|"
    r"position|index|epoch|split|profile|attack_kind|strength|"
    r"serialization|hash|coordinate|router_id)",
    re.IGNORECASE,
```

_Additional matches omitted: 89_

## Source evidence: Role Labels

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:14` (match at line 17)

```python
    physical_port_mask bool    [16,10]
    y_attack           float32 scalar
    y_attacker_count   int64   scalar
    y_source           float32 [16]
    y_transit          float32 [16]
    y_victim           float32 [16]
    y_attack_path      float32 [16]
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:15` (match at line 18)

```python
    y_attack           float32 scalar
    y_attacker_count   int64   scalar
    y_source           float32 [16]
    y_transit          float32 [16]
    y_victim           float32 [16]
    y_attack_path      float32 [16]
    role_mask          uint8/bool [16]
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:16` (match at line 19)

```python
    y_attacker_count   int64   scalar
    y_source           float32 [16]
    y_transit          float32 [16]
    y_victim           float32 [16]
    y_attack_path      float32 [16]
    role_mask          uint8/bool [16]

```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:18` (match at line 21)

```python
    y_transit          float32 [16]
    y_victim           float32 [16]
    y_attack_path      float32 [16]
    role_mask          uint8/bool [16]

No identifiers, metadata, edge_index, coordinates, run lengths, or window
positions are returned.
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:50` (match at line 53)

```python
    "physical_port_mask",
    "y_attack",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim",
    "y_attack_path",
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:51` (match at line 54)

```python
    "y_attack",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim",
    "y_attack_path",
    "role_mask",
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:52` (match at line 55)

```python
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim",
    "y_attack_path",
    "role_mask",
}
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:54` (match at line 57)

```python
    "y_transit",
    "y_victim",
    "y_attack_path",
    "role_mask",
}


```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:378` (match at line 381)

```python
            "y_attacker_count": (
                run["y_attacker_count"][entry.target].clone()
            ),
            "y_source": run["y_source"][entry.target].clone(),
            "y_transit": run["y_transit"][entry.target].clone(),
            "y_victim": run["y_victim"][entry.target].clone(),
            "y_attack_path": (
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:379` (match at line 382)

```python
                run["y_attacker_count"][entry.target].clone()
            ),
            "y_source": run["y_source"][entry.target].clone(),
            "y_transit": run["y_transit"][entry.target].clone(),
            "y_victim": run["y_victim"][entry.target].clone(),
            "y_attack_path": (
                run["y_attack_path"][entry.target].clone()
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:380` (match at line 383)

```python
            ),
            "y_source": run["y_source"][entry.target].clone(),
            "y_transit": run["y_transit"][entry.target].clone(),
            "y_victim": run["y_victim"][entry.target].clone(),
            "y_attack_path": (
                run["y_attack_path"][entry.target].clone()
            ),
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:384` (match at line 387)

```python
            "y_attack_path": (
                run["y_attack_path"][entry.target].clone()
            ),
            "role_mask": run["role_mask"][entry.target].clone(),
        }

        if set(item) != MODEL_ITEM_KEYS:
```

### `scripts/failure_analysis/analyze_v3_matched_runs.py:737` (match at line 740)

```python
                "match_type": exact_match_type,
                "factor_match_score": flags["factor_match_score"],
                "factor_match_fraction": flags["factor_match_fraction"],
                "probability_source_available": True,
                "feature_tensor_loaded": False,
                "training_prediction_available": False,
            }
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:213` (match at line 216)

```python
                "flow_side": flow_side,
                "direction": direction,
                "is_directional": bool(direction),
                "taxonomy_source": "fixed_stage7b_corrected_taxonomy",
            }
        )

```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:2152` (match at line 2155)

```python
    feature_cols_path = dataset_root / "feature_cols.npy"

    alignment_path = table_directory / "stage7_sample_alignment.csv"
    run_inventory_path = table_directory / "stage7_run_inventory.csv"
    selected_inventory_path = (
        table_directory / "stage7_selected_run_inventory.csv"
    )
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:2153` (match at line 2156)

```python

    alignment_path = table_directory / "stage7_sample_alignment.csv"
    run_inventory_path = table_directory / "stage7_run_inventory.csv"
    selected_inventory_path = (
        table_directory / "stage7_selected_run_inventory.csv"
    )
    topology_path = table_directory / "stage7_topology_manifest.csv"
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:2156` (match at line 2159)

```python
    selected_inventory_path = (
        table_directory / "stage7_selected_run_inventory.csv"
    )
    topology_path = table_directory / "stage7_topology_manifest.csv"
    stage7a_summary_path = table_directory / "stage7a_alignment_summary.json"

    output_paths = {
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:2157` (match at line 2160)

```python
        table_directory / "stage7_selected_run_inventory.csv"
    )
    topology_path = table_directory / "stage7_topology_manifest.csv"
    stage7a_summary_path = table_directory / "stage7a_alignment_summary.json"

    output_paths = {
        "corrected_taxonomy": table_directory / "stage7b_corrected_feature_taxonomy.csv",
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:2191` (match at line 2194)

```python
        x_path,
        feature_cols_path,
        alignment_path,
        run_inventory_path,
        selected_inventory_path,
        topology_path,
        stage7a_summary_path,
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:2192` (match at line 2195)

```python
        feature_cols_path,
        alignment_path,
        run_inventory_path,
        selected_inventory_path,
        topology_path,
        stage7a_summary_path,
    ]
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:2193` (match at line 2196)

```python
        alignment_path,
        run_inventory_path,
        selected_inventory_path,
        topology_path,
        stage7a_summary_path,
    ]
    for path in required_paths:
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:2194` (match at line 2197)

```python
        run_inventory_path,
        selected_inventory_path,
        topology_path,
        stage7a_summary_path,
    ]
    for path in required_paths:
        if not path.is_file():
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:2241` (match at line 2244)

```python
        },
    )
    run_inventory = pd.read_csv(
        run_inventory_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:2247` (match at line 2250)

```python
        dtype={"run_id": str, "split": str},
    )
    selected_inventory = pd.read_csv(
        selected_inventory_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:2253` (match at line 2256)

```python
        dtype={"run_id": str, "split": str},
    )
    topology = pd.read_csv(
        topology_path,
        keep_default_na=False,
        low_memory=False,
    )
```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:2257` (match at line 2260)

```python
        keep_default_na=False,
        low_memory=False,
    )
    stage7a_summary = json.loads(stage7a_summary_path.read_text())

    if not stage7a_summary.get("overall_success", False):
        raise RuntimeError("Stage 7A summary does not report overall success.")
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2476` (match at line 2479)

```python

    x_path = dataset_root / "x.npy"
    alignment_path = table_dir / "stage7_sample_alignment.csv"
    run_inventory_path = table_dir / "stage7_run_inventory.csv"
    selected_inventory_path = table_dir / "stage7_selected_run_inventory.csv"
    taxonomy_path = table_dir / "stage7b_corrected_feature_taxonomy.csv"
    run_summary_path = table_dir / "stage7b_run_feature_summary.csv"
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2477` (match at line 2480)

```python
    x_path = dataset_root / "x.npy"
    alignment_path = table_dir / "stage7_sample_alignment.csv"
    run_inventory_path = table_dir / "stage7_run_inventory.csv"
    selected_inventory_path = table_dir / "stage7_selected_run_inventory.csv"
    taxonomy_path = table_dir / "stage7b_corrected_feature_taxonomy.csv"
    run_summary_path = table_dir / "stage7b_run_feature_summary.csv"
    router_summary_path = table_dir / "stage7b_selected_router_feature_summary.csv"
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2478` (match at line 2481)

```python
    alignment_path = table_dir / "stage7_sample_alignment.csv"
    run_inventory_path = table_dir / "stage7_run_inventory.csv"
    selected_inventory_path = table_dir / "stage7_selected_run_inventory.csv"
    taxonomy_path = table_dir / "stage7b_corrected_feature_taxonomy.csv"
    run_summary_path = table_dir / "stage7b_run_feature_summary.csv"
    router_summary_path = table_dir / "stage7b_selected_router_feature_summary.csv"
    temporal_summary_path = table_dir / "stage7b_selected_temporal_feature_summary.csv"
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2479` (match at line 2482)

```python
    run_inventory_path = table_dir / "stage7_run_inventory.csv"
    selected_inventory_path = table_dir / "stage7_selected_run_inventory.csv"
    taxonomy_path = table_dir / "stage7b_corrected_feature_taxonomy.csv"
    run_summary_path = table_dir / "stage7b_run_feature_summary.csv"
    router_summary_path = table_dir / "stage7b_selected_router_feature_summary.csv"
    temporal_summary_path = table_dir / "stage7b_selected_temporal_feature_summary.csv"
    spatial_summary_path = table_dir / "stage7b_selected_spatial_concentration.csv"
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2480` (match at line 2483)

```python
    selected_inventory_path = table_dir / "stage7_selected_run_inventory.csv"
    taxonomy_path = table_dir / "stage7b_corrected_feature_taxonomy.csv"
    run_summary_path = table_dir / "stage7b_run_feature_summary.csv"
    router_summary_path = table_dir / "stage7b_selected_router_feature_summary.csv"
    temporal_summary_path = table_dir / "stage7b_selected_temporal_feature_summary.csv"
    spatial_summary_path = table_dir / "stage7b_selected_spatial_concentration.csv"
    stage7b_summary_path = table_dir / "stage7b_summary.json"
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2481` (match at line 2484)

```python
    taxonomy_path = table_dir / "stage7b_corrected_feature_taxonomy.csv"
    run_summary_path = table_dir / "stage7b_run_feature_summary.csv"
    router_summary_path = table_dir / "stage7b_selected_router_feature_summary.csv"
    temporal_summary_path = table_dir / "stage7b_selected_temporal_feature_summary.csv"
    spatial_summary_path = table_dir / "stage7b_selected_spatial_concentration.csv"
    stage7b_summary_path = table_dir / "stage7b_summary.json"

```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2482` (match at line 2485)

```python
    run_summary_path = table_dir / "stage7b_run_feature_summary.csv"
    router_summary_path = table_dir / "stage7b_selected_router_feature_summary.csv"
    temporal_summary_path = table_dir / "stage7b_selected_temporal_feature_summary.csv"
    spatial_summary_path = table_dir / "stage7b_selected_spatial_concentration.csv"
    stage7b_summary_path = table_dir / "stage7b_summary.json"

    outputs = {
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2483` (match at line 2486)

```python
    router_summary_path = table_dir / "stage7b_selected_router_feature_summary.csv"
    temporal_summary_path = table_dir / "stage7b_selected_temporal_feature_summary.csv"
    spatial_summary_path = table_dir / "stage7b_selected_spatial_concentration.csv"
    stage7b_summary_path = table_dir / "stage7b_summary.json"

    outputs = {
        "training_run_reference": table_dir / "stage7c1_v2_training_run_reference.csv",
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2536` (match at line 2539)

```python
    required = [
        x_path,
        alignment_path,
        run_inventory_path,
        selected_inventory_path,
        taxonomy_path,
        run_summary_path,
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2537` (match at line 2540)

```python
        x_path,
        alignment_path,
        run_inventory_path,
        selected_inventory_path,
        taxonomy_path,
        run_summary_path,
        router_summary_path,
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2538` (match at line 2541)

```python
        alignment_path,
        run_inventory_path,
        selected_inventory_path,
        taxonomy_path,
        run_summary_path,
        router_summary_path,
        temporal_summary_path,
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2539` (match at line 2542)

```python
        run_inventory_path,
        selected_inventory_path,
        taxonomy_path,
        run_summary_path,
        router_summary_path,
        temporal_summary_path,
        spatial_summary_path,
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2540` (match at line 2543)

```python
        selected_inventory_path,
        taxonomy_path,
        run_summary_path,
        router_summary_path,
        temporal_summary_path,
        spatial_summary_path,
        stage7b_summary_path,
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2541` (match at line 2544)

```python
        taxonomy_path,
        run_summary_path,
        router_summary_path,
        temporal_summary_path,
        spatial_summary_path,
        stage7b_summary_path,
    ]
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2542` (match at line 2545)

```python
        run_summary_path,
        router_summary_path,
        temporal_summary_path,
        spatial_summary_path,
        stage7b_summary_path,
    ]
    for path in required:
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2543` (match at line 2546)

```python
        router_summary_path,
        temporal_summary_path,
        spatial_summary_path,
        stage7b_summary_path,
    ]
    for path in required:
        if not path.is_file():
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2579` (match at line 2582)

```python
        dtype={"run_id": str, "split": str, "sample_id": str},
    )
    run_inventory = pd.read_csv(
        run_inventory_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2585` (match at line 2588)

```python
        dtype={"run_id": str, "split": str},
    )
    selected_inventory = pd.read_csv(
        selected_inventory_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2590` (match at line 2593)

```python
        low_memory=False,
        dtype={"run_id": str, "split": str},
    )
    taxonomy = pd.read_csv(taxonomy_path, keep_default_na=False)
    run_summary = pd.read_csv(
        run_summary_path,
        keep_default_na=False,
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2592` (match at line 2595)

```python
    )
    taxonomy = pd.read_csv(taxonomy_path, keep_default_na=False)
    run_summary = pd.read_csv(
        run_summary_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2598` (match at line 2601)

```python
        dtype={"run_id": str, "split": str},
    )
    selected_router_summary = pd.read_csv(
        router_summary_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2604` (match at line 2607)

```python
        dtype={"run_id": str, "split": str},
    )
    selected_temporal_summary = pd.read_csv(
        temporal_summary_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2610` (match at line 2613)

```python
        dtype={"run_id": str, "split": str},
    )
    selected_spatial_summary = pd.read_csv(
        spatial_summary_path,
        keep_default_na=False,
        low_memory=False,
        dtype={"run_id": str, "split": str},
```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2615` (match at line 2618)

```python
        low_memory=False,
        dtype={"run_id": str, "split": str},
    )
    stage7b_summary = json.loads(stage7b_summary_path.read_text())
    if not stage7b_summary.get("overall_success", False):
        raise RuntimeError("Stage 7B did not report overall success.")

```

### `scripts/failure_analysis/analyze_v3_stage7c1_training_coverage_v2.py:2622` (match at line 2625)

```python
    taxonomy_valid = (
        len(taxonomy) == EXPECTED_FEATURES
        and taxonomy["feature_index"].tolist() == list(range(EXPECTED_FEATURES))
        and taxonomy["taxonomy_source"].eq(
            "fixed_stage7b_corrected_taxonomy"
        ).all()
    )
```

### `scripts/failure_analysis/audit_v3_feature_provenance_h1b0.py:331` (match at line 334)

```python
        "model_core_rtl_work_authorized": True,
        "next_stage": "H1B.1_AUTHORITATIVE_SOURCE_RESOLUTION",
    }
    summary_path = output_dir / "h1b0_provenance_inventory.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
```

### `scripts/failure_analysis/audit_v3_feature_provenance_h1b0.py:332` (match at line 335)

```python
        "next_stage": "H1B.1_AUTHORITATIVE_SOURCE_RESOLUTION",
    }
    summary_path = output_dir / "h1b0_provenance_inventory.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
```

### `scripts/failure_analysis/audit_v3_feature_provenance_h1b0.py:398` (match at line 401)

```python
        "resolution_template="
        f"{reports / 'feature_provenance_resolution_template.csv'}"
    )
    print(f"inventory_report={summary_path}")
    print(f"artifact_hashes={hashes_path}")
    print(f"release_manifest={release_path}")
    print("next_stage=H1B.1_AUTHORITATIVE_SOURCE_RESOLUTION")
```

### `scripts/failure_analysis/close_and_freeze_chrono_b1.py:349` (match at line 352)

```python
        "- `b1_closure_summary.md`",
        "",
    ]
    summary_path = output_dir / "b1_closure_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    closure["closure_outputs"] = {
```

### `scripts/failure_analysis/close_and_freeze_chrono_b1.py:350` (match at line 353)

```python
        "",
    ]
    summary_path = output_dir / "b1_closure_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    closure["closure_outputs"] = {
        "manifest": str(manifest_path),
```

### `scripts/failure_analysis/close_and_freeze_chrono_b1.py:355` (match at line 358)

```python
    closure["closure_outputs"] = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
    }
    manifest_path.write_text(
```

### `scripts/failure_analysis/close_and_freeze_chrono_b1.py:356` (match at line 359)

```python
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
    }
    manifest_path.write_text(
        json.dumps(closure, indent=2) + "\n",
```

### `scripts/failure_analysis/close_and_freeze_chrono_b1.py:375` (match at line 378)

```python
    print(f"git_branch={git_branch['stdout']}")
    print(f"git_status_entries={len(git_status['stdout'].splitlines()) if git_status['stdout'] else 0}")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")
    print(f"inventory={inventory_csv}")


```

### `scripts/failure_analysis/close_c1_and_freeze_final_v3.py:287` (match at line 290)

```python
    ]

    output_dir.mkdir(parents=True, exist_ok=False)
    inventory_path = output_dir / "final_v3_artifact_hashes.csv"
    write_csv(inventory_path, inventory)

    git_head = run(["git", "rev-parse", "HEAD"], repo_root)
```

### `scripts/failure_analysis/close_c1_and_freeze_final_v3.py:288` (match at line 291)

```python

    output_dir.mkdir(parents=True, exist_ok=False)
    inventory_path = output_dir / "final_v3_artifact_hashes.csv"
    write_csv(inventory_path, inventory)

    git_head = run(["git", "rev-parse", "HEAD"], repo_root)
    git_branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
```

### `scripts/failure_analysis/close_c1_and_freeze_final_v3.py:303` (match at line 306)

```python
        "final_decision": final_decision,
        "key_metrics": key_metrics,
        "artifact_inventory": {
            "path": str(inventory_path),
            "sha256": sha256(inventory_path),
            "file_count": len(inventory),
        },
```

### `scripts/failure_analysis/close_c1_and_freeze_final_v3.py:304` (match at line 307)

```python
        "key_metrics": key_metrics,
        "artifact_inventory": {
            "path": str(inventory_path),
            "sha256": sha256(inventory_path),
            "file_count": len(inventory),
        },
        "repository": {
```

### `scripts/failure_analysis/close_c1_and_freeze_final_v3.py:383` (match at line 386)

```python
more strongly toward feature semantics, representation ambiguity, and scenario
distribution mismatch.
"""
    summary_path = output_dir / "final_v3_architecture_freeze.md"
    summary_path.write_text(summary, encoding="utf-8")

    print("C1.6 FINAL V3 ARCHITECTURE FREEZE: PASS")
```

### `scripts/failure_analysis/close_c1_and_freeze_final_v3.py:384` (match at line 387)

```python
distribution mismatch.
"""
    summary_path = output_dir / "final_v3_architecture_freeze.md"
    summary_path.write_text(summary, encoding="utf-8")

    print("C1.6 FINAL V3 ARCHITECTURE FREEZE: PASS")
    print("status=V3_ARCHITECTURE_FROZEN")
```

### `scripts/failure_analysis/close_c1_and_freeze_final_v3.py:408` (match at line 411)

```python
    )
    print("next_phase=V3.1 raw-trace and feature-semantics audit")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")
    print(f"inventory={inventory_path}")


```

### `scripts/failure_analysis/close_c1_and_freeze_final_v3.py:409` (match at line 412)

```python
    print("next_phase=V3.1 raw-trace and feature-semantics audit")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")
    print(f"inventory={inventory_path}")


if __name__ == "__main__":
```

### `scripts/failure_analysis/create_v3_a1_rtl_h0_package.py:77` (match at line 80)

```python
    out = args.output_dir.resolve()

    checkpoint_path = model_dir / "best_model.pt"
    summary_path = model_dir / "summary.json"
    history_path = model_dir / "history.json"
    splits_path = model_dir / "splits.npz"
    x_path = data_dir / "x.npy"
```

### `scripts/failure_analysis/create_v3_a1_rtl_h0_package.py:78` (match at line 81)

```python

    checkpoint_path = model_dir / "best_model.pt"
    summary_path = model_dir / "summary.json"
    history_path = model_dir / "history.json"
    splits_path = model_dir / "splits.npz"
    x_path = data_dir / "x.npy"
    edge_path = data_dir / "edge_index.npy"
```

### `scripts/failure_analysis/create_v3_a1_rtl_h0_package.py:85` (match at line 88)

```python
    features_path = data_dir / "feature_cols.npy"

    required = [
        source, checkpoint_path, summary_path, splits_path,
        x_path, edge_path, features_path
    ]
    for path in required:
```

### `scripts/failure_analysis/create_v3_a1_rtl_h0_package.py:147` (match at line 150)

```python
    copies = [
        (source, dirs["model"] / "train_temporal_gcn_v3.py"),
        (checkpoint_path, dirs["model"] / "best_model.pt"),
        (summary_path, dirs["model"] / "summary.json"),
        (splits_path, dirs["model"] / "splits.npz"),
        (edge_path, dirs["adjacency"] / "edge_index.npy"),
        (features_path, dirs["features"] / "feature_cols.npy"),
```

### `scripts/failure_analysis/create_v3_a1_rtl_h0_package.py:152` (match at line 155)

```python
        (edge_path, dirs["adjacency"] / "edge_index.npy"),
        (features_path, dirs["features"] / "feature_cols.npy"),
    ]
    if history_path.is_file():
        copies.append((history_path, dirs["model"] / "history.json"))
    for src, dst in copies:
        shutil.copy2(src, dst)
```

### `scripts/failure_analysis/create_v3_a1_rtl_h0_package.py:153` (match at line 156)

```python
        (features_path, dirs["features"] / "feature_cols.npy"),
    ]
    if history_path.is_file():
        copies.append((history_path, dirs["model"] / "history.json"))
    for src, dst in copies:
        shutil.copy2(src, dst)

```

### `scripts/failure_analysis/dry_run_chrono_b1_sampler.py:318` (match at line 321)

```python
        },
    }

    summary_path = output_dir / "sampler_dry_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
```

### `scripts/failure_analysis/dry_run_chrono_b1_sampler.py:319` (match at line 322)

```python
    }

    summary_path = output_dir / "sampler_dry_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
```

### `scripts/failure_analysis/evaluate_chrono_c1_validation_gate.py:1014` (match at line 1017)

```python
    a1_model_dir = args.a1_model_dir.resolve()
    c1_model_dir = args.c1_model_dir.resolve()
    spec_path = args.c1_spec.resolve()
    integrity_path = args.c1_integrity_report.resolve()
    output_dir = args.output_dir.resolve()

    required_paths = {
```

### `scripts/failure_analysis/evaluate_chrono_c1_validation_gate.py:1026` (match at line 1029)

```python
        "a1_splits": a1_model_dir / "splits.npz",
        "c1_splits": c1_model_dir / "splits.npz",
        "c1_spec": spec_path,
        "c1_integrity_report": integrity_path,
    }
    for label, path in required_paths.items():
        if label == "data_dir":
```

### `scripts/failure_analysis/evaluate_chrono_c1_validation_gate.py:1043` (match at line 1046)

```python
        raise SystemExit("STOP: batch size must be positive.")

    specification = load_json(spec_path)
    integrity = load_json(integrity_path)

    prerequisite_checks = {
        "spec_stage_is_c1_0": specification.get("stage") == "C1.0",
```

### `scripts/failure_analysis/evaluate_chrono_g1_validation_gate.py:577` (match at line 580)

```python
    g1_source = args.g1_source.resolve()
    a1_model_dir = args.a1_model_dir.resolve()
    g1_model_dir = args.g1_model_dir.resolve()
    training_integrity_path = args.g1_training_integrity.resolve()
    output_dir = args.output_dir.resolve()

    a1_checkpoint = a1_model_dir / "best_model.pt"
```

### `scripts/failure_analysis/evaluate_chrono_g1_validation_gate.py:594` (match at line 597)

```python
        "g1_checkpoint": g1_checkpoint,
        "a1_splits": a1_splits,
        "g1_splits": g1_splits,
        "g1_training_integrity": training_integrity_path,
        "y_graph": data_dir / "y_graph.npy",
        "y_node": data_dir / "y_node.npy",
        "run_id": data_dir / "run_id.npy",
```

_Additional matches omitted: 804_

## Source evidence: Endpoint Semantics

### `scripts/failure_analysis/resolve_v3_feature_builders_h1b1a.py:340` (match at line 343)

```python
                        "same_as_likely_candidate": False,
                    })

    # Include H1B.0R evidence rows corresponding to the likely sources.
    likely_paths = {str(Path(row["file_path"]).resolve()) for row in likely_rows}
    inherited_evidence = [
        row for row in evidence_rows
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:292` (match at line 295)

```python
    node_logit = module.logit(data.node_prob)
    count_logprob = np.log(np.clip(count_prob, 1e-8, 1.0))

    endpoints, graph_agg = module.aggregate_array(
        graph_logit,
        segments,
        horizon,
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:298` (match at line 301)

```python
        horizon,
        module.uniform_weights(horizon),
    )
    node_endpoints, node_agg = module.aggregate_array(
        node_logit,
        segments,
        horizon,
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:304` (match at line 307)

```python
        horizon,
        module.uniform_weights(horizon),
    )
    count_endpoints, count_agg = module.aggregate_array(
        count_logprob,
        segments,
        horizon,
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:311` (match at line 314)

```python
        module.uniform_weights(horizon),
    )
    if not (
        np.array_equal(endpoints, node_endpoints)
        and np.array_equal(endpoints, count_endpoints)
    ):
        raise RuntimeError("aggregation endpoint mismatch")
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:312` (match at line 315)

```python
    )
    if not (
        np.array_equal(endpoints, node_endpoints)
        and np.array_equal(endpoints, count_endpoints)
    ):
        raise RuntimeError("aggregation endpoint mismatch")

```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:314` (match at line 317)

```python
        np.array_equal(endpoints, node_endpoints)
        and np.array_equal(endpoints, count_endpoints)
    ):
        raise RuntimeError("aggregation endpoint mismatch")

    graph_score = module.sigmoid(graph_agg)
    node_score = module.sigmoid(node_agg)
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:332` (match at line 335)

```python

    final_pred = apply_persistence(
        gated_pred,
        data.run_index[endpoints],
        str(policy.get("persistence_rule", "none")),
    )

```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:337` (match at line 340)

```python
    )

    return {
        "endpoints": endpoints,
        "graph_score": graph_score,
        "node_score": node_score,
        "count_score": count_score,
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:371` (match at line 374)

```python
    expected = base_gidx[horizon - 1 :]
    if not np.array_equal(expected, decision_gidx):
        raise RuntimeError(
            f"feature endpoint alignment failed for run={run_index}, "
            f"router={router}, H={horizon}"
        )

```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:633` (match at line 636)

```python
    feature_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

    def get_run_indices(policy_name: str, target_run: int) -> np.ndarray:
        endpoints = policy_cache[policy_name]["endpoints"]
        return np.flatnonzero(run_index[endpoints] == target_run)

    def get_feature_sequence(
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:634` (match at line 637)

```python

    def get_run_indices(policy_name: str, target_run: int) -> np.ndarray:
        endpoints = policy_cache[policy_name]["endpoints"]
        return np.flatnonzero(run_index[endpoints] == target_run)

    def get_feature_sequence(
        policy_name: str,
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:646` (match at line 649)

```python
        if key in feature_cache:
            return feature_cache[key]
        positions = get_run_indices(policy_name, target_run)
        endpoints = policy_cache[policy_name]["endpoints"][positions]
        gidx = global_index[endpoints]
        value = rolling_feature_sequence(
            x_memmap,
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:647` (match at line 650)

```python
            return feature_cache[key]
        positions = get_run_indices(policy_name, target_run)
        endpoints = policy_cache[policy_name]["endpoints"][positions]
        gidx = global_index[endpoints]
        value = rolling_feature_sequence(
            x_memmap,
            dataset_run_index,
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:711` (match at line 714)

```python

        cache = policy_cache[policy_name]
        positions = get_run_indices(policy_name, target_run)
        endpoints = cache["endpoints"][positions]
        final_pred = cache["final_pred"][positions]
        gated_pred = cache["gated_pred"][positions]
        raw_pred = cache["raw_pred"][positions]
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:728` (match at line 731)

```python
                f"{hard['false_isolation_decisions']}"
            )

        decision_gidx = global_index[endpoints]
        decision_epochs = np.asarray(dataset_end_epoch[decision_gidx])
        feature_mean, feature_peak = get_feature_sequence(
            policy_name,
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:808` (match at line 811)

```python
                    if interval_id_by_decision[i] >= 0
                    else ""
                ),
                "endpoint_archive_row": int(endpoints[i]),
                "global_index": int(decision_gidx[i]),
                "end_epoch": int(decision_epochs[i]),
                "raw_graph_probability": float(graph_prob[endpoints[i]]),
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:811` (match at line 814)

```python
                "endpoint_archive_row": int(endpoints[i]),
                "global_index": int(decision_gidx[i]),
                "end_epoch": int(decision_epochs[i]),
                "raw_graph_probability": float(graph_prob[endpoints[i]]),
                "aggregated_graph_probability": float(graph_score[i]),
                "graph_gate_open": bool(graph_gate[i]),
                "raw_router_probability": float(node_prob[endpoints[i], router]),
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:814` (match at line 817)

```python
                "raw_graph_probability": float(graph_prob[endpoints[i]]),
                "aggregated_graph_probability": float(graph_score[i]),
                "graph_gate_open": bool(graph_gate[i]),
                "raw_router_probability": float(node_prob[endpoints[i], router]),
                "aggregated_router_probability": float(node_score[i, router]),
                "router_rank": int(target_ranks[i]),
                "node_threshold_pass": bool(raw_pred[i, router]),
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:821` (match at line 824)

```python
                "graph_gated_router_selected": bool(gated_pred[i, router]),
                "policy_output_router_selected": bool(target_selected[i]),
                "policy_selected_set": selected_set,
                "raw_count_prediction": int(np.argmax(count_prob[endpoints[i]])),
                "aggregated_count_prediction": int(np.argmax(count_score[i])),
                "true_attacker_count": int(true_count[endpoints[i]]),
                "neighbor_score_mean": float(neighbor_mean[i]),
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:823` (match at line 826)

```python
                "policy_selected_set": selected_set,
                "raw_count_prediction": int(np.argmax(count_prob[endpoints[i]])),
                "aggregated_count_prediction": int(np.argmax(count_score[i])),
                "true_attacker_count": int(true_count[endpoints[i]]),
                "neighbor_score_mean": float(neighbor_mean[i]),
                "neighbor_score_max": float(neighbor_max[i]),
                "target_minus_neighbor_max": float(
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:844` (match at line 847)

```python
        attack_refs_same_profile: list[int] = []
        attack_refs_any_profile: list[int] = []

        full_endpoints = cache["endpoints"]
        full_final_pred = cache["final_pred"]
        for candidate_run in unique_test_runs:
            if candidate_run == target_run or candidate_run >= len(runs):
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:857` (match at line 860)

```python
            c_label = int(candidate_meta.get("label", 0) or 0)
            c_attackers = parse_nodes(candidate_meta.get("attackers"))
            c_positions = np.flatnonzero(
                run_index[full_endpoints] == candidate_run
            )
            if c_positions.size == 0:
                continue
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:949` (match at line 952)

```python
                ref_positions = get_run_indices(policy_name, reference_run)
                if ref_positions.size == 0:
                    continue
                ref_endpoints = cache["endpoints"][ref_positions]
                ref_mean, ref_peak = get_feature_sequence(
                    policy_name,
                    reference_run,
```

### `scripts/failure_analysis/v4_a3_11_c_hard_normal_trace.py:968` (match at line 971)

```python
                    + 1
                )
                if cohort == "true_attacker_decisions":
                    keep = y_node[ref_endpoints, router] == 1
                else:
                    keep = np.ones(ref_positions.size, dtype=bool)
                if not np.any(keep):
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:335` (match at line 338)

```python
    graph_logit = module.logit(data.graph_prob)
    node_logit = module.logit(data.node_prob)

    endpoints, graph_agg = module.aggregate_array(
        graph_logit,
        segments,
        horizon,
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:341` (match at line 344)

```python
        horizon,
        module.uniform_weights(horizon),
    )
    node_endpoints, node_agg = module.aggregate_array(
        node_logit,
        segments,
        horizon,
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:347` (match at line 350)

```python
        horizon,
        module.uniform_weights(horizon),
    )
    if not np.array_equal(endpoints, node_endpoints):
        raise RuntimeError("graph/node endpoint mismatch")

    return endpoints, module.sigmoid(graph_agg), module.sigmoid(node_agg)
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:348` (match at line 351)

```python
        module.uniform_weights(horizon),
    )
    if not np.array_equal(endpoints, node_endpoints):
        raise RuntimeError("graph/node endpoint mismatch")

    return endpoints, module.sigmoid(graph_agg), module.sigmoid(node_agg)

```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:350` (match at line 353)

```python
    if not np.array_equal(endpoints, node_endpoints):
        raise RuntimeError("graph/node endpoint mismatch")

    return endpoints, module.sigmoid(graph_agg), module.sigmoid(node_agg)


def evaluate_localization_policy(
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:361` (match at line 364)

```python
    deoverlap_stride: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    horizon = int(policy["horizon"])
    endpoints, graph_score, node_score = aggregate_scores(
        module,
        data,
        dense_segments,
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:379` (match at line 382)

```python

    pred = apply_persistence(
        pred,
        data.run_index[endpoints],
        str(policy.get("persistence_rule", "none")),
    )

```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:384` (match at line 387)

```python
    )

    metrics = module.node_metrics(
        data.y_graph[endpoints],
        data.y_node[endpoints],
        pred,
        data.attacker_count[endpoints],
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:385` (match at line 388)

```python

    metrics = module.node_metrics(
        data.y_graph[endpoints],
        data.y_node[endpoints],
        pred,
        data.attacker_count[endpoints],
        np.sum(pred, axis=1).astype(np.int64),
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:387` (match at line 390)

```python
        data.y_graph[endpoints],
        data.y_node[endpoints],
        pred,
        data.attacker_count[endpoints],
        np.sum(pred, axis=1).astype(np.int64),
    )
    metrics.update(
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:392` (match at line 395)

```python
    )
    metrics.update(
        module.operational_metrics(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            data.y_node[endpoints],
            pred,
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:393` (match at line 396)

```python
    metrics.update(
        module.operational_metrics(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            data.y_node[endpoints],
            pred,
        )
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:394` (match at line 397)

```python
        module.operational_metrics(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            data.y_node[endpoints],
            pred,
        )
    )
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:399` (match at line 402)

```python
        )
    )
    metrics["graph_gate_attack_open_rate"] = float(
        np.mean(graph_score[data.y_graph[endpoints] == 1]
                >= float(policy["graph_threshold"]))
    )
    metrics["graph_gate_normal_open_rate"] = float(
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:403` (match at line 406)

```python
                >= float(policy["graph_threshold"]))
    )
    metrics["graph_gate_normal_open_rate"] = float(
        np.mean(graph_score[data.y_graph[endpoints] == 0]
                >= float(policy["graph_threshold"]))
    )

```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:407` (match at line 410)

```python
                >= float(policy["graph_threshold"]))
    )

    return endpoints, pred, metrics


def main() -> int:
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:513` (match at line 516)

```python
    hard_normal_rows: list[dict[str, Any]] = []

    # P0 graph alert.
    p0_endpoints, p0_graph_score, _ = aggregate_scores(
        module,
        data,
        dense_segments,
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:524` (match at line 527)

```python
        p0_graph_score >= float(p0["graph_threshold"])
    ).astype(np.int8)
    p0_metrics = module.binary_metrics(
        data.y_graph[p0_endpoints],
        p0_pred,
    )
    p0_metrics.update(
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:529` (match at line 532)

```python
    )
    p0_metrics.update(
        graph_operational_metrics(
            data.run_index[p0_endpoints],
            data.y_graph[p0_endpoints],
            p0_pred,
        )
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:530` (match at line 533)

```python
    p0_metrics.update(
        graph_operational_metrics(
            data.run_index[p0_endpoints],
            data.y_graph[p0_endpoints],
            p0_pred,
        )
    )
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:547` (match at line 550)

```python
    }

    for name, policy in named_policies.items():
        endpoints, pred, metrics = evaluate_localization_policy(
            module,
            data,
            dense_segments,
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:560` (match at line 563)

```python
        }

        rows = hard_normal_run_rows(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            pred,
        )
```

### `scripts/failure_analysis/v4_a3_11_frozen_test_transfer.py:561` (match at line 564)

```python

        rows = hard_normal_run_rows(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            pred,
        )
        for row in rows:
```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:291` (match at line 294)

```python
    prediction_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}

    for horizon in HORIZONS:
        endpoints, graph_agg = module.aggregate_array(
            graph_logit,
            segments,
            horizon,
```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:297` (match at line 300)

```python
            horizon,
            module.uniform_weights(horizon),
        )
        node_endpoints, node_agg = module.aggregate_array(
            node_logit,
            segments,
            horizon,
```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:303` (match at line 306)

```python
            horizon,
            module.uniform_weights(horizon),
        )
        if not np.array_equal(endpoints, node_endpoints):
            raise RuntimeError("graph/node endpoint mismatch")

        graph_score = module.sigmoid(graph_agg)
```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:304` (match at line 307)

```python
            module.uniform_weights(horizon),
        )
        if not np.array_equal(endpoints, node_endpoints):
            raise RuntimeError("graph/node endpoint mismatch")

        graph_score = module.sigmoid(graph_agg)
        node_score = module.sigmoid(node_agg)
```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:309` (match at line 312)

```python
        graph_score = module.sigmoid(graph_agg)
        node_score = module.sigmoid(node_agg)

        y_graph = data.y_graph[endpoints]
        y_node = data.y_node[endpoints]
        true_count = data.attacker_count[endpoints]
        run_index = data.run_index[endpoints]
```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:310` (match at line 313)

```python
        node_score = module.sigmoid(node_agg)

        y_graph = data.y_graph[endpoints]
        y_node = data.y_node[endpoints]
        true_count = data.attacker_count[endpoints]
        run_index = data.run_index[endpoints]

```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:311` (match at line 314)

```python

        y_graph = data.y_graph[endpoints]
        y_node = data.y_node[endpoints]
        true_count = data.attacker_count[endpoints]
        run_index = data.run_index[endpoints]

        evidence_span_epochs = horizon * args.deoverlap_stride
```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:312` (match at line 315)

```python
        y_graph = data.y_graph[endpoints]
        y_node = data.y_node[endpoints]
        true_count = data.attacker_count[endpoints]
        run_index = data.run_index[endpoints]

        evidence_span_epochs = horizon * args.deoverlap_stride

```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:396` (match at line 399)

```python
                        graph_threshold,
                        rule,
                    )
                    prediction_cache[key] = (endpoints, final_pred)

        print(
            f"completed H={horizon}; cumulative_candidates={len(candidates)}",
```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:451` (match at line 454)

```python
            float(selected["graph_threshold"]),
            str(selected["persistence_rule"]),
        )
        endpoints, pred = prediction_cache[key]
        hard_rows = hard_normal_run_rows(
            data.run_index[endpoints],
            data.y_graph[endpoints],
```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:453` (match at line 456)

```python
        )
        endpoints, pred = prediction_cache[key]
        hard_rows = hard_normal_run_rows(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            pred,
        )
```

### `scripts/failure_analysis/v4_a3_11_l2_graph_gated_refinement.py:454` (match at line 457)

```python
        endpoints, pred = prediction_cache[key]
        hard_rows = hard_normal_run_rows(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            pred,
        )
    write_csv(args.output_dir / "selected_policy_hard_normal_runs.csv", hard_rows)
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:202` (match at line 205)

```python
    graph_logit = module.logit(data.graph_prob)
    node_logit = module.logit(data.node_prob)

    endpoints, graph_agg = module.aggregate_array(
        graph_logit,
        segments,
        horizon,
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:216` (match at line 219)

```python
    )

    return {
        "graph_mean_logit": (endpoints, module.sigmoid(graph_agg)),
        "node_mean_logit": (endpoints, module.sigmoid(node_agg)),
    }

```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:217` (match at line 220)

```python

    return {
        "graph_mean_logit": (endpoints, module.sigmoid(graph_agg)),
        "node_mean_logit": (endpoints, module.sigmoid(node_agg)),
    }


```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:230` (match at line 233)

```python
    graph_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache = build_cache(module, data, segments, horizon)
    endpoints, node_score = cache["node_mean_logit"]
    pred = (node_score >= node_threshold).astype(np.int8)

    if graph_threshold is not None:
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:234` (match at line 237)

```python
    pred = (node_score >= node_threshold).astype(np.int8)

    if graph_threshold is not None:
        graph_endpoints, graph_score = cache["graph_mean_logit"]
        if not np.array_equal(graph_endpoints, endpoints):
            raise RuntimeError("graph/node endpoint mismatch")
        pred[graph_score < graph_threshold] = 0
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:235` (match at line 238)

```python

    if graph_threshold is not None:
        graph_endpoints, graph_score = cache["graph_mean_logit"]
        if not np.array_equal(graph_endpoints, endpoints):
            raise RuntimeError("graph/node endpoint mismatch")
        pred[graph_score < graph_threshold] = 0

```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:236` (match at line 239)

```python
    if graph_threshold is not None:
        graph_endpoints, graph_score = cache["graph_mean_logit"]
        if not np.array_equal(graph_endpoints, endpoints):
            raise RuntimeError("graph/node endpoint mismatch")
        pred[graph_score < graph_threshold] = 0

    metrics = module.node_metrics(
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:240` (match at line 243)

```python
        pred[graph_score < graph_threshold] = 0

    metrics = module.node_metrics(
        data.y_graph[endpoints],
        data.y_node[endpoints],
        pred,
        data.attacker_count[endpoints],
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:241` (match at line 244)

```python

    metrics = module.node_metrics(
        data.y_graph[endpoints],
        data.y_node[endpoints],
        pred,
        data.attacker_count[endpoints],
        np.sum(pred, axis=1).astype(np.int64),
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:243` (match at line 246)

```python
        data.y_graph[endpoints],
        data.y_node[endpoints],
        pred,
        data.attacker_count[endpoints],
        np.sum(pred, axis=1).astype(np.int64),
    )
    metrics.update(
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:248` (match at line 251)

```python
    )
    metrics.update(
        module.operational_metrics(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            data.y_node[endpoints],
            pred,
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:249` (match at line 252)

```python
    metrics.update(
        module.operational_metrics(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            data.y_node[endpoints],
            pred,
        )
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:250` (match at line 253)

```python
        module.operational_metrics(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            data.y_node[endpoints],
            pred,
        )
    )
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:254` (match at line 257)

```python
            pred,
        )
    )
    return endpoints, pred, metrics


def main() -> int:
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:343` (match at line 346)

```python
    # Alert policy.
    alert = policies["P0_FAST_ALERT"]
    cache = build_cache(module, data, segments, alert["horizon"])
    endpoints, graph_score = cache["graph_mean_logit"]
    graph_pred = (graph_score >= alert["graph_threshold"]).astype(np.int8)
    alert_metrics = module.binary_metrics(
        data.y_graph[endpoints],
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:346` (match at line 349)

```python
    endpoints, graph_score = cache["graph_mean_logit"]
    graph_pred = (graph_score >= alert["graph_threshold"]).astype(np.int8)
    alert_metrics = module.binary_metrics(
        data.y_graph[endpoints],
        graph_pred,
    )
    alert_metrics.update(
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:351` (match at line 354)

```python
    )
    alert_metrics.update(
        graph_operational_metrics(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            graph_pred,
        )
```

### `scripts/failure_analysis/v4_a3_11_latency_family_audit.py:352` (match at line 355)

```python
    alert_metrics.update(
        graph_operational_metrics(
            data.run_index[endpoints],
            data.y_graph[endpoints],
            graph_pred,
        )
    )
```

### `scripts/failure_analysis/v4_a3_11_validation_search.py:448` (match at line 451)

```python
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate along each segment. weights are ordered oldest -> newest.
    Returns endpoint indices and aggregated values.
    """
    values = np.asarray(arr)
    if values.ndim == 1:
```

### `scripts/failure_analysis/v4_a3_11_validation_search.py:457` (match at line 460)

```python
    else:
        squeeze = False

    endpoint_parts: list[np.ndarray] = []
    output_parts: list[np.ndarray] = []

    w = np.asarray(weights, dtype=np.float64)
```

_Additional matches omitted: 94_

## Source evidence: Port Order

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:11` (match at line 14)

```python

Returned model/target dictionary:
    x                  float32 [16,58,32]
    physical_port_mask bool    [16,10]
    y_attack           float32 scalar
    y_attacker_count   int64   scalar
    y_source           float32 [16]
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:47` (match at line 50)

```python

MODEL_ITEM_KEYS = {
    "x",
    "physical_port_mask",
    "y_attack",
    "y_attacker_count",
    "y_source",
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:78` (match at line 81)

```python
        ) from exc


def derive_corrected_physical_port_mask(
    topology: dict[str, Any],
) -> torch.Tensor:
    """
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:83` (match at line 86)

```python
) -> torch.Tensor:
    """
    Derive Boolean [16,10] in this exact order:
      input : local,north,east,south,west
      output: local,north,east,south,west

    Frozen A2-R2 coordinate semantics:
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:84` (match at line 87)

```python
    """
    Derive Boolean [16,10] in this exact order:
      input : local,north,east,south,west
      output: local,north,east,south,west

    Frozen A2-R2 coordinate semantics:
      vertical   = coord[0]
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:117` (match at line 120)

```python
    west = horizontal > horizontal_min

    one_side = torch.stack(
        (local, north, east, south, west),
        dim=1,
    )
    mask = torch.cat((one_side, one_side), dim=1)
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:196` (match at line 199)

```python
        )
        if not isinstance(topology, dict):
            raise TypeError("topology.pt payload must be a dictionary")
        self.physical_port_mask = (
            derive_corrected_physical_port_mask(topology)
        )

```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:197` (match at line 200)

```python
        if not isinstance(topology, dict):
            raise TypeError("topology.pt payload must be a dictionary")
        self.physical_port_mask = (
            derive_corrected_physical_port_mask(topology)
        )

        self._index: list[IndexEntry] = []
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:371` (match at line 374)

```python

        item = {
            "x": x,
            "physical_port_mask": (
                self.physical_port_mask.clone()
            ),
            "y_attack": run["y_attack"][entry.target].clone(),
```

### `src/data/v5_p2_pair_aligned_primary58_dataset.py:372` (match at line 375)

```python
        item = {
            "x": x,
            "physical_port_mask": (
                self.physical_port_mask.clone()
            ),
            "y_attack": run["y_attack"][entry.target].clone(),
            "y_attacker_count": (
```

### `src/models/v5_frozen_b3_conv1d_only.py:54` (match at line 57)

```python
    ------
    x:
        float tensor [B,16,58,32]
    physical_port_mask:
        bool/float tensor [B,16,10]

    Outputs
```

### `src/models/v5_frozen_b3_conv1d_only.py:133` (match at line 136)

```python
    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(
```

### `src/models/v5_frozen_b3_conv1d_only.py:140` (match at line 143)

```python
                f"x shape={tuple(x.shape)}, expected [B,16,58,32]"
            )
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
```

### `src/models/v5_frozen_b3_conv1d_only.py:141` (match at line 144)

```python
            )
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError(
```

### `src/models/v5_frozen_b3_conv1d_only.py:142` (match at line 145)

```python
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError(
                "physical_port_mask must have shape [B,16,10]"
```

### `src/models/v5_frozen_b3_conv1d_only.py:145` (match at line 148)

```python
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError(
                "physical_port_mask must have shape [B,16,10]"
            )

        batch_size, num_nodes, num_features, time_steps = x.shape
```

### `src/models/v5_frozen_b3_conv1d_only.py:168` (match at line 171)

```python
        node_input = torch.cat(
            (
                temporal_embedding,
                physical_port_mask.to(
                    dtype=temporal_embedding.dtype
                ),
            ),
```

### `src/models/v5_p2_b3_conv1d_only_count4.py:64` (match at line 67)

```python
    ------
    x:
        float tensor [B,16,58,32]
    physical_port_mask:
        bool/float tensor [B,16,10]

    Outputs
```

### `src/models/v5_p2_b3_conv1d_only_count4.py:147` (match at line 150)

```python
    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            x.ndim != 4
```

### `src/models/v5_p2_b3_conv1d_only_count4.py:158` (match at line 161)

```python
                "expected [B,16,58,32]"
            )
        if (
            physical_port_mask.ndim != 3
            or tuple(
                physical_port_mask.shape[1:]
            ) != (16, 10)
```

### `src/models/v5_p2_b3_conv1d_only_count4.py:160` (match at line 163)

```python
        if (
            physical_port_mask.ndim != 3
            or tuple(
                physical_port_mask.shape[1:]
            ) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
```

### `src/models/v5_p2_b3_conv1d_only_count4.py:162` (match at line 165)

```python
            or tuple(
                physical_port_mask.shape[1:]
            ) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError(
                "physical_port_mask must have shape "
```

### `src/models/v5_p2_b3_conv1d_only_count4.py:165` (match at line 168)

```python
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError(
                "physical_port_mask must have shape "
                "[B,16,10]"
            )

```

### `src/models/v5_p2_b3_conv1d_only_count4.py:191` (match at line 194)

```python
        node_input = torch.cat(
            (
                temporal_embedding,
                physical_port_mask.to(
                    dtype=temporal_embedding.dtype
                ),
            ),
```

### `src/models/v5_p2_task_d_full_multitask_count4.py:86` (match at line 89)

```python
    def encode_pre_graph(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
```

### `src/models/v5_p2_task_d_full_multitask_count4.py:91` (match at line 94)

```python
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
```

### `src/models/v5_p2_task_d_full_multitask_count4.py:92` (match at line 95)

```python
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")
```

### `src/models/v5_p2_task_d_full_multitask_count4.py:93` (match at line 96)

```python
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")

```

### `src/models/v5_p2_task_d_full_multitask_count4.py:95` (match at line 98)

```python
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")

        batch_size, num_nodes, num_features, time_steps = x.shape
        y = x.reshape(batch_size * num_nodes, num_features, time_steps)
```

### `src/models/v5_p2_task_d_full_multitask_count4.py:106` (match at line 109)

```python
        node_input = torch.cat(
            (
                temporal_embedding,
                physical_port_mask.to(dtype=temporal_embedding.dtype),
            ),
            dim=-1,
        )
```

### `src/models/v5_p2_task_d_full_multitask_count4.py:115` (match at line 118)

```python
    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_pre_graph(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
```

### `src/models/v5_p2_task_d_full_multitask_count4.py:117` (match at line 120)

```python
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_pre_graph(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
        flat = node_embedding.reshape(batch_size * num_nodes, embedding_dim)
        edges = batched_edge_index(
```

### `scripts/failure_analysis/align_v3_stage7a.py:106` (match at line 109)

```python
    "node_preds",
]

DIRECTION_WORDS = ["local", "north", "east", "south", "west"]


# ---------------------------------------------------------------------------
```

### `scripts/failure_analysis/align_v3_stage7a.py:1309` (match at line 1312)

```python
            "true_graph": graph_labels,
            "true_attacker_count": true_attacker_count,
            "is_stage3_exported": False,
            "stage3_export_order": -1,
        }
    )
    sample_alignment.loc[stage3_indices, "is_stage3_exported"] = True
```

### `scripts/failure_analysis/align_v3_stage7a.py:1315` (match at line 1318)

```python
    sample_alignment.loc[stage3_indices, "is_stage3_exported"] = True
    sample_alignment.loc[
        stage3_indices,
        "stage3_export_order",
    ] = np.arange(len(stage3_indices), dtype=np.int64)
    sample_alignment = add_temporal_positions(sample_alignment)

```

### `scripts/failure_analysis/analyze_v3_stage7b_features.py:44` (match at line 47)

```python
    "ifd_out_norm_west",
]

DIRECTIONS = ["local", "north", "east", "south", "west"]

HARD_NORMAL_RUN = "N-3-7-8-12-Pmixed-R18-V3"
NORMAL_TEST_CONTROL = "N-1-6-9-14-Pmixed-R17-V3"
```

### `scripts/failure_analysis/audit_v3_feature_provenance_h1b0.py:29` (match at line 32)

```python
    "in_count_norm_", "out_count_norm_", "ifd_in_norm_", "ifd_out_norm_",
    "clip", "clipping", "normalize", "normalization", "divisor", "scale",
    "missing", "invalid", "valid_port", "flit_observed", "gap_observed",
    "raw_ifd", "preclip", "pre_clip", "port_order", "direction",
    "local", "north", "east", "south", "west",
]

```

### `scripts/failure_analysis/audit_v3_feature_provenance_h1b0.py:30` (match at line 33)

```python
    "clip", "clipping", "normalize", "normalization", "divisor", "scale",
    "missing", "invalid", "valid_port", "flit_observed", "gap_observed",
    "raw_ifd", "preclip", "pre_clip", "port_order", "direction",
    "local", "north", "east", "south", "west",
]


```

### `scripts/failure_analysis/audit_v3_stage8d_final_consistency.py:96` (match at line 99)

```python
    },
}

PORTS = ["local", "north", "east", "south", "west"]


@dataclass
```

### `scripts/failure_analysis/v4/run_v4_a3_complete_analysis.py:42` (match at line 45)

```python
    EXPERIMENT_DESIGNATION,
    MODEL_NAME,
    NUM_ROUTERS,
    PORT_ORDER,
    REGIONAL_EMBEDDING_DIM,
    atomic_csv_dump,
    atomic_json_dump,
```

### `scripts/failure_analysis/v4/run_v4_a3_complete_analysis.py:904` (match at line 907)

```python
        out / "valid_port_mask_specification.json",
        {
            "standalone_4x4_mask": physical_mask.tolist(),
            "port_order": list(PORT_ORDER),
            "router_order": "router_id = row * 4 + column",
            "hierarchical_rule": "physical validity is computed from global mesh coordinates and is distinct from intra-region GCN adjacency",
        },
```

### `scripts/failure_analysis/v4/run_v4_pretraining_diagnostic.py:519` (match at line 522)

```python

def router_direction_valid(router, direction, mesh_cols=4):
    row, col = divmod(router, mesh_cols)
    return {"north": row > 0, "east": col < mesh_cols - 1, "south": row < mesh_cols - 1, "west": col > 0}[direction]


class RateAccumulator:
```

### `scripts/failure_analysis/v4_a3_15_directional_source_transit.py:22` (match at line 25)

```python
  1  ifd_out_norm
  2  input_flit_count_norm
  3  output_flit_count_norm
  4:9   in_count_norm_[local,north,east,south,west]
  9:14  out_count_norm_[local,north,east,south,west]
  14:19 ifd_in_norm_[local,north,east,south,west]
  19:24 ifd_out_norm_[local,north,east,south,west]
```

### `scripts/failure_analysis/v4_a3_15_directional_source_transit.py:23` (match at line 26)

```python
  2  input_flit_count_norm
  3  output_flit_count_norm
  4:9   in_count_norm_[local,north,east,south,west]
  9:14  out_count_norm_[local,north,east,south,west]
  14:19 ifd_in_norm_[local,north,east,south,west]
  19:24 ifd_out_norm_[local,north,east,south,west]

```

### `scripts/failure_analysis/v4_a3_15_directional_source_transit.py:24` (match at line 27)

```python
  3  output_flit_count_norm
  4:9   in_count_norm_[local,north,east,south,west]
  9:14  out_count_norm_[local,north,east,south,west]
  14:19 ifd_in_norm_[local,north,east,south,west]
  19:24 ifd_out_norm_[local,north,east,south,west]

Derived quantities are explicitly labeled as proxies. They do not replace
```

### `scripts/failure_analysis/v4_a3_15_directional_source_transit.py:25` (match at line 28)

```python
  4:9   in_count_norm_[local,north,east,south,west]
  9:14  out_count_norm_[local,north,east,south,west]
  14:19 ifd_in_norm_[local,north,east,south,west]
  19:24 ifd_out_norm_[local,north,east,south,west]

Derived quantities are explicitly labeled as proxies. They do not replace
missing dataset semantics such as a direct source/transit/valid-port label.
```

### `scripts/v4/a4a/v4_a4a_slot_model.py:86` (match at line 89)

```python
        self,
        x: torch.Tensor,
        a_hat: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 8, 24):
            raise ValueError(
```

### `scripts/v4/a4a/v4_a4a_slot_model.py:110` (match at line 113)

```python
            dim=1,
        ).reshape(batch_size, NUM_ROUTERS, 32)

        if tuple(physical_port_mask.shape) == (NUM_ROUTERS, 5):
            mask = physical_port_mask.unsqueeze(0).expand(
                batch_size,
                -1,
```

### `scripts/v4/a4a/v4_a4a_slot_model.py:111` (match at line 114)

```python
        ).reshape(batch_size, NUM_ROUTERS, 32)

        if tuple(physical_port_mask.shape) == (NUM_ROUTERS, 5):
            mask = physical_port_mask.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
```

### `scripts/v4/a4a/v4_a4a_slot_model.py:116` (match at line 119)

```python
                -1,
                -1,
            )
        elif tuple(physical_port_mask.shape) == (
            batch_size,
            NUM_ROUTERS,
            5,
```

### `scripts/v4/a4a/v4_a4a_slot_model.py:121` (match at line 124)

```python
            NUM_ROUTERS,
            5,
        ):
            mask = physical_port_mask
        else:
            raise ValueError(
                "physical_port_mask must be [16,5] or [B,16,5], "
```

### `scripts/v4/a4a/v4_a4a_slot_model.py:124` (match at line 127)

```python
            mask = physical_port_mask
        else:
            raise ValueError(
                "physical_port_mask must be [16,5] or [B,16,5], "
                f"got {tuple(physical_port_mask.shape)}"
            )

```

### `scripts/v4/a4a/v4_a4a_slot_model.py:125` (match at line 128)

```python
        else:
            raise ValueError(
                "physical_port_mask must be [16,5] or [B,16,5], "
                f"got {tuple(physical_port_mask.shape)}"
            )

        local_input = torch.cat(
```

### `scripts/v4/a4a/v4_a4a_slot_model.py:266` (match at line 269)

```python
        self,
        x: torch.Tensor,
        a_hat: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            representation = self.encoder(
```

### `scripts/v4/a4a/v4_a4a_slot_model.py:272` (match at line 275)

```python
            representation = self.encoder(
                x,
                a_hat,
                physical_port_mask,
            )

        decoded = self.slot_decoder(
```

### `scripts/v4/common/v4_a3_sourcepreserve.py:45` (match at line 48)

```python
NODE_FUSED_DIM = 32
REGIONAL_EMBEDDING_DIM = 64
COUNT_CLASSES = 5
PORT_ORDER = ("local", "north", "east", "south", "west")
PRIMARY_SEED = 7


```

### `scripts/v4/common/v4_a3_sourcepreserve.py:313` (match at line 316)

```python
    mesh_rows: int = MESH_ROWS,
    mesh_cols: int = MESH_COLS,
) -> torch.Tensor:
    """Return [routers,5] in local,north,east,south,west order."""
    if mesh_rows * mesh_cols != NUM_ROUTERS:
        raise ValueError("This V4-A3 model expects exactly 16 routers")
    mask = torch.zeros((NUM_ROUTERS, 5), dtype=torch.float32)
```

### `scripts/v4/common/v4_a3_sourcepreserve.py:398` (match at line 401)

```python
        self,
        x: torch.Tensor,
        a_hat: torch.Tensor,
        physical_port_mask: torch.Tensor,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (NUM_ROUTERS, WINDOW_EPOCHS, INPUT_FEATURES):
```

### `scripts/v4/common/v4_a3_sourcepreserve.py:413` (match at line 416)

```python
        temporal_pooled = torch.cat([temporal_mean, temporal_max], dim=1).reshape(
            batch_size, NUM_ROUTERS, TEMPORAL_DIM * 2
        )
        if tuple(physical_port_mask.shape) == (NUM_ROUTERS, 5):
            mask = physical_port_mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif tuple(physical_port_mask.shape) == (batch_size, NUM_ROUTERS, 5):
            mask = physical_port_mask
```

### `scripts/v4/common/v4_a3_sourcepreserve.py:414` (match at line 417)

```python
            batch_size, NUM_ROUTERS, TEMPORAL_DIM * 2
        )
        if tuple(physical_port_mask.shape) == (NUM_ROUTERS, 5):
            mask = physical_port_mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif tuple(physical_port_mask.shape) == (batch_size, NUM_ROUTERS, 5):
            mask = physical_port_mask
        else:
```

### `scripts/v4/common/v4_a3_sourcepreserve.py:415` (match at line 418)

```python
        )
        if tuple(physical_port_mask.shape) == (NUM_ROUTERS, 5):
            mask = physical_port_mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif tuple(physical_port_mask.shape) == (batch_size, NUM_ROUTERS, 5):
            mask = physical_port_mask
        else:
            raise ValueError(
```

### `scripts/v4/common/v4_a3_sourcepreserve.py:416` (match at line 419)

```python
        if tuple(physical_port_mask.shape) == (NUM_ROUTERS, 5):
            mask = physical_port_mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif tuple(physical_port_mask.shape) == (batch_size, NUM_ROUTERS, 5):
            mask = physical_port_mask
        else:
            raise ValueError(
                f"physical_port_mask must be [16,5] or [B,16,5], got {tuple(physical_port_mask.shape)}"
```

### `scripts/v4/common/v4_a3_sourcepreserve.py:419` (match at line 422)

```python
            mask = physical_port_mask
        else:
            raise ValueError(
                f"physical_port_mask must be [16,5] or [B,16,5], got {tuple(physical_port_mask.shape)}"
            )
        local_input = torch.cat([temporal_pooled, mask.to(temporal_pooled.dtype)], dim=2)
        h_local = F.relu(self.local_projection(local_input))
```

### `scripts/v4/common/v4_a3_sourcepreserve.py:473` (match at line 476)

```python
            "padding": list(model.temporal_conv.padding),
        },
        "temporal_readout": "concatenate mean and max over 8 epochs",
        "physical_valid_port_mask": list(PORT_ORDER),
        "local_projection": [model.local_projection.in_features, model.local_projection.out_features],
        "gcn_count": gcn_instances,
        "gcn": [model.gcn.linear.in_features, model.gcn.linear.out_features],
```

### `scripts/v4/train/train_v4_a3_sourcepreserve.py:357` (match at line 360)

```python
        "pass": all(value is True or isinstance(value, str) for value in checks.values()),
        "checks": checks,
        "headers": headers,
        "physical_port_mask": mask.tolist(),
        "port_order": ["local", "north", "east", "south", "west"],
        "mask_report": mask_report,
        "adjacency_shape": list(adjacency.shape),
```

### `scripts/v4/train/train_v4_a3_sourcepreserve.py:358` (match at line 361)

```python
        "checks": checks,
        "headers": headers,
        "physical_port_mask": mask.tolist(),
        "port_order": ["local", "north", "east", "south", "west"],
        "mask_report": mask_report,
        "adjacency_shape": list(adjacency.shape),
        "data_dir": str(data_dir),
```

### `scripts/v5/baselines/run_v5_p0_b7b_multi_seed.py:156` (match at line 159)

```python
    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
        edge_index: torch.Tensor,
        return_diagnostics: bool = False,
    ):
```

### `scripts/v5/baselines/run_v5_p0_b7b_multi_seed.py:162` (match at line 165)

```python
    ):
        if x.ndim != 4 or x.shape[1:] != (16, 58, 32):
            raise ValueError(f"expected x [B,16,58,32], got {tuple(x.shape)}")
        if physical_port_mask.ndim != 3 or physical_port_mask.shape[1:] != (16, 10):
            raise ValueError(
                "expected physical_port_mask [B,16,10], got "
                f"{tuple(physical_port_mask.shape)}"
```

### `scripts/v5/baselines/run_v5_p0_b7b_multi_seed.py:164` (match at line 167)

```python
            raise ValueError(f"expected x [B,16,58,32], got {tuple(x.shape)}")
        if physical_port_mask.ndim != 3 or physical_port_mask.shape[1:] != (16, 10):
            raise ValueError(
                "expected physical_port_mask [B,16,10], got "
                f"{tuple(physical_port_mask.shape)}"
            )
        self.verify_edge_index(edge_index)
```

### `scripts/v5/baselines/run_v5_p0_b7b_multi_seed.py:165` (match at line 168)

```python
        if physical_port_mask.ndim != 3 or physical_port_mask.shape[1:] != (16, 10):
            raise ValueError(
                "expected physical_port_mask [B,16,10], got "
                f"{tuple(physical_port_mask.shape)}"
            )
        self.verify_edge_index(edge_index)
        batch_size, router_count, feature_count, window = x.shape
```

### `scripts/v5/baselines/run_v5_p0_b7b_multi_seed.py:174` (match at line 177)

```python
        for block in self.temporal_blocks:
            temporal = block(temporal)
        node_temporal = temporal[..., -1].reshape(batch_size, router_count, -1)
        mask = physical_port_mask.to(device=x.device, dtype=x.dtype)
        h_local = self.node_post(torch.cat([node_temporal, mask], dim=-1))
        graph_repr = torch.cat(
            [h_local.mean(dim=1), h_local.max(dim=1).values],
```

### `scripts/v5/baselines/run_v5_p0_b7b_multi_seed.py:304` (match at line 307)

```python
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                x=batch["x"].to(device=device, dtype=torch.float32),
                physical_port_mask=batch["physical_port_mask"].to(device=device),
                edge_index=batch["edge_index"].to(device=device, dtype=torch.long),
            )
            loss, components = module.compute_loss(
```

### `scripts/v5/baselines/run_v5_p0_b7b_multi_seed.py:620` (match at line 623)

```python
    sample = train_dataset[0]
    if tuple(sample["x"].shape) != (16, 58, 32):
        raise RuntimeError(f"unexpected x shape: {tuple(sample['x'].shape)}")
    if sample["physical_port_mask"].dtype != torch.bool:
        raise RuntimeError("physical_port_mask is not Boolean")
    module.validate_mesh_edge_index(sample["edge_index"])

```

### `scripts/v5/baselines/run_v5_p0_b7b_multi_seed.py:621` (match at line 624)

```python
    if tuple(sample["x"].shape) != (16, 58, 32):
        raise RuntimeError(f"unexpected x shape: {tuple(sample['x'].shape)}")
    if sample["physical_port_mask"].dtype != torch.bool:
        raise RuntimeError("physical_port_mask is not Boolean")
    module.validate_mesh_edge_index(sample["edge_index"])

    stats_loader = DataLoader(
```

### `scripts/v5/baselines/train_v5_p0_b1_static_final_epoch_mlp.py:11` (match at line 14)

```python
Frozen learned inputs
---------------------
- PRIMARY58 x at x[..., -1] only;
- topology-derived raw Boolean physical_port_mask.

Explicitly unused
-----------------
```

### `scripts/v5/baselines/train_v5_p0_b1_static_final_epoch_mlp.py:279` (match at line 282)

```python
    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B,N,F,W], got {tuple(x.shape)}")
```

### `scripts/v5/baselines/train_v5_p0_b1_static_final_epoch_mlp.py:288` (match at line 291)

```python
                "expected x trailing shape [16,58,32], got "
                f"{tuple(x.shape[1:])}"
            )
        if physical_port_mask.ndim != 3:
            raise ValueError(
                "physical_port_mask must have shape [B,16,10], got "
                f"{tuple(physical_port_mask.shape)}"
```

### `scripts/v5/baselines/train_v5_p0_b1_static_final_epoch_mlp.py:290` (match at line 293)

```python
            )
        if physical_port_mask.ndim != 3:
            raise ValueError(
                "physical_port_mask must have shape [B,16,10], got "
                f"{tuple(physical_port_mask.shape)}"
            )
        if physical_port_mask.shape[1:] != (16, 10):
```

### `scripts/v5/baselines/train_v5_p0_b1_static_final_epoch_mlp.py:291` (match at line 294)

```python
        if physical_port_mask.ndim != 3:
            raise ValueError(
                "physical_port_mask must have shape [B,16,10], got "
                f"{tuple(physical_port_mask.shape)}"
            )
        if physical_port_mask.shape[1:] != (16, 10):
            raise ValueError(
```

### `scripts/v5/baselines/train_v5_p0_b1_static_final_epoch_mlp.py:293` (match at line 296)

```python
                "physical_port_mask must have shape [B,16,10], got "
                f"{tuple(physical_port_mask.shape)}"
            )
        if physical_port_mask.shape[1:] != (16, 10):
            raise ValueError(
                "expected mask trailing shape [16,10], got "
                f"{tuple(physical_port_mask.shape[1:])}"
```

_Additional matches omitted: 295_

## Security boundary

- Dataset directories scanned: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- L0 locked: **false**
