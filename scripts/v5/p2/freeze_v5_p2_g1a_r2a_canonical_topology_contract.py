#!/usr/bin/env python3
"""
V5 P2-G1A-R2A Canonical Static Topology Contract

R2 proved that the P2 Conv1D dataset/loader does not store edge_index.
That is not a data error: the frozen B3 Conv1D model never required a graph.

This append-only stage:
- preserves G1A, G1A-R1, and G1A-R2 HOLD outputs;
- audits non-test project sources for topology/numbering evidence;
- freezes an explicit canonical row-major 4x4 physical mesh topology;
- generates edge_index, adjacency, coordinates, and neighbor-table artifacts;
- authorizes G1 graph-baseline implementation to consume those artifacts.

No model training, checkpoint loading, prediction-cache access, test access,
quantization, RTL generation, or Legal NoC decoder work is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGE = "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT"
COMPLETE = f"{STAGE}_COMPLETE"

EXPECTED_G0_PROTOCOL_SHA = (
    "22e3d9828edaba9323a0c5206806ea93c149a239b8d9043ad79129de76882def"
)
EXPECTED_R2_HOLD = (
    "V5_P2_G1A_R2_STATIC_EDGE_INDEX_RESOLUTION_HOLD"
)

TEXT_SUFFIXES = {
    ".py", ".sh", ".json", ".md", ".txt", ".yaml", ".yml", ".csv"
}
SKIP_DIRS = {
    ".git", ".venv", "__pycache__", "node_modules", "models"
}
FORBIDDEN_TOKENS = {
    "p2_b6",
    "b6_one_shot",
    "blind_test",
    "test_evaluation",
    "p2_c0",
    "p2_c1",
    "p2_l5",
}

SEARCH_PATTERNS = [
    re.compile(r"\b4\s*[x×]\s*4\b", re.IGNORECASE),
    re.compile(r"\bnum_nodes\b", re.IGNORECASE),
    re.compile(r"\bnum_routers\b", re.IGNORECASE),
    re.compile(r"\bedge_index\b", re.IGNORECASE),
    re.compile(r"\brow[_ -]?major\b", re.IGNORECASE),
    re.compile(r"\brouter[_ -]?id\b", re.IGNORECASE),
    re.compile(r"\bdivmod\s*\(", re.IGNORECASE),
    re.compile(r"\bcoord(?:inate)?[_ -]?[01xy]\b", re.IGNORECASE),
    re.compile(r"\bmesh\b", re.IGNORECASE),
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def path_is_forbidden(path: Path) -> bool:
    lowered = "/".join(part.lower() for part in path.parts)
    return any(token in lowered for token in FORBIDDEN_TOKENS)


def safe_walk(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in SKIP_DIRS
            and not path_is_forbidden(current_path / directory)
        ]
        for filename in files:
            path = current_path / filename
            if not path_is_forbidden(path):
                yield path


def audit_topology_evidence(repo: Path, loader: Path) -> dict[str, Any]:
    roots = [
        loader,
        repo / "src",
        repo / "scripts" / "v5",
        repo / "reports" / "v5",
        repo / "data" / "processed" / "v5",
    ]

    files_scanned = 0
    matching_files = 0
    snippets: list[dict[str, Any]] = []

    seen: set[str] = set()
    candidate_files: list[Path] = []

    for root in roots:
        if root.is_file():
            candidate_files.append(root)
        elif root.is_dir():
            candidate_files.extend(safe_walk(root))

    for path in candidate_files:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)

        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > 10 * 1024 * 1024:
            continue

        files_scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        file_matches = 0
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not any(pattern.search(line) for pattern in SEARCH_PATTERNS):
                continue
            snippets.append(
                {
                    "path": str(path.resolve()),
                    "line": line_number,
                    "text": line.strip()[:1000],
                }
            )
            file_matches += 1
            if len(snippets) >= 300:
                break

        if file_matches:
            matching_files += 1
        if len(snippets) >= 300:
            break

    return {
        "files_scanned": files_scanned,
        "matching_files": matching_files,
        "snippet_count": len(snippets),
        "snippets": snippets,
    }


def build_canonical_mesh():
    rows = 4
    cols = 4
    num_nodes = rows * cols

    coordinates = np.zeros((num_nodes, 2), dtype=np.int64)
    directed_edges: list[tuple[int, int]] = []
    neighbors: dict[str, list[int]] = {}

    for router in range(num_nodes):
        row, col = divmod(router, cols)
        coordinates[router] = [row, col]

        router_neighbors: list[int] = []
        if row > 0:
            router_neighbors.append((row - 1) * cols + col)
        if col > 0:
            router_neighbors.append(row * cols + (col - 1))
        if col + 1 < cols:
            router_neighbors.append(row * cols + (col + 1))
        if row + 1 < rows:
            router_neighbors.append((row + 1) * cols + col)

        router_neighbors = sorted(router_neighbors)
        neighbors[str(router)] = router_neighbors
        directed_edges.extend(
            (router, destination)
            for destination in router_neighbors
        )

    directed_edges = sorted(set(directed_edges))
    edge_index = np.asarray(directed_edges, dtype=np.int64).T

    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.uint8)
    for source, destination in directed_edges:
        adjacency[source, destination] = 1

    return coordinates, edge_index, adjacency, neighbors


def validate_canonical(
    coordinates: np.ndarray,
    edge_index: np.ndarray,
    adjacency: np.ndarray,
    neighbors: dict[str, list[int]],
) -> list[str]:
    failures: list[str] = []

    if coordinates.shape != (16, 2):
        failures.append(
            f"coordinate shape {coordinates.shape} != (16,2)"
        )
    if edge_index.shape != (2, 48):
        failures.append(
            f"edge_index shape {edge_index.shape} != (2,48)"
        )
    if adjacency.shape != (16, 16):
        failures.append(
            f"adjacency shape {adjacency.shape} != (16,16)"
        )
    if int(adjacency.sum()) != 48:
        failures.append(
            f"adjacency sum {int(adjacency.sum())} != 48"
        )
    if np.any(np.diag(adjacency) != 0):
        failures.append("physical adjacency contains self-loops")
    if not np.array_equal(adjacency, adjacency.T):
        failures.append("physical adjacency is not symmetric")
    if edge_index.min() != 0 or edge_index.max() != 15:
        failures.append("edge_index router range is not 0..15")
    if len({tuple(edge) for edge in edge_index.T.tolist()}) != 48:
        failures.append("edge_index does not contain 48 unique edges")

    for source, destination in edge_index.T.tolist():
        source_row, source_col = coordinates[source].tolist()
        destination_row, destination_col = coordinates[destination].tolist()
        manhattan = (
            abs(source_row - destination_row)
            + abs(source_col - destination_col)
        )
        if manhattan != 1:
            failures.append(
                f"non-adjacent edge {source}->{destination}"
            )
            break

    expected_degrees = {
        0: 2, 3: 2, 12: 2, 15: 2,
        1: 3, 2: 3, 4: 3, 7: 3,
        8: 3, 11: 3, 13: 3, 14: 3,
        5: 4, 6: 4, 9: 4, 10: 4,
    }
    for router, expected_degree in expected_degrees.items():
        actual_degree = len(neighbors[str(router)])
        if actual_degree != expected_degree:
            failures.append(
                f"router {router} degree {actual_degree} "
                f"!= {expected_degree}"
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--g0-dir", type=Path, required=True)
    parser.add_argument("--r2-hold-dir", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    g0_dir = args.g0_dir.expanduser().resolve()
    r2_hold_dir = args.r2_hold_dir.expanduser().resolve()
    loader = args.loader.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: output already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    g0_lock_path = (
        g0_dir
        / "V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL_LOCK.json"
    )
    g0_operator_path = (
        g0_dir
        / "V5_P2_G0_GRAPH_OPERATOR_CONTRACTS.json"
    )
    r2_report_path = (
        r2_hold_dir
        / "V5_P2_G1A_R2_STATIC_EDGE_INDEX_RESOLUTION.json"
    )
    r2_lock_path = (
        r2_hold_dir
        / "V5_P2_G1A_R2_STATIC_EDGE_INDEX_RESOLUTION_LOCK.json"
    )
    r2_hold_marker = (
        r2_hold_dir
        / "V5_P2_G1A_R2_STATIC_EDGE_INDEX_RESOLUTION_HOLD"
    )

    for path in (
        g0_lock_path,
        g0_operator_path,
        r2_report_path,
        r2_lock_path,
        r2_hold_marker,
        loader,
    ):
        if not path.is_file():
            failures.append(f"missing prerequisite: {path}")

    g0_operator: dict[str, Any] = {}
    r2_report: dict[str, Any] = {}

    if not failures:
        g0_lock = json.loads(
            g0_lock_path.read_text(encoding="utf-8")
        )
        g0_operator = json.loads(
            g0_operator_path.read_text(encoding="utf-8")
        )
        r2_report = json.loads(
            r2_report_path.read_text(encoding="utf-8")
        )
        r2_lock = json.loads(
            r2_lock_path.read_text(encoding="utf-8")
        )

        if g0_lock.get("protocol_sha256") != EXPECTED_G0_PROTOCOL_SHA:
            failures.append("G0 protocol SHA changed")
        if (
            r2_hold_marker.read_text(encoding="utf-8").strip()
            != EXPECTED_R2_HOLD
        ):
            failures.append("R2 HOLD marker changed")
        if r2_lock.get("report_sha256") != sha256_file(r2_report_path):
            failures.append("R2 report SHA mismatch")
        if r2_report.get("status") != "HOLD":
            failures.append("R2 report is not HOLD")
        if (
            r2_report.get("security_boundary", {})
            .get("test_tensors_deserialized")
            is not False
        ):
            failures.append("R2 test-data boundary changed")

        common = g0_operator.get("common_architecture", {})
        frozen_edge_description = str(common.get("edge_index", ""))
        if "4x4" not in frozen_edge_description:
            failures.append(
                "G0 operator contract no longer freezes a 4x4 topology"
            )
        if "48" not in frozen_edge_description:
            failures.append(
                "G0 operator contract no longer freezes 48 directed edges"
            )

    evidence = (
        audit_topology_evidence(repo, loader)
        if not failures
        else {}
    )

    coordinates, edge_index, adjacency, neighbors = (
        build_canonical_mesh()
    )
    failures.extend(
        validate_canonical(
            coordinates,
            edge_index,
            adjacency,
            neighbors,
        )
    )

    coords_path = (
        output_dir
        / "V5_P2_G1A_R2A_ROUTER_COORDINATES_ROW_MAJOR.npy"
    )
    edge_path = (
        output_dir
        / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy"
    )
    adjacency_path = (
        output_dir
        / "V5_P2_G1A_R2A_CANONICAL_ADJACENCY_MATRIX.npy"
    )
    neighbors_path = (
        output_dir
        / "V5_P2_G1A_R2A_CANONICAL_NEIGHBOR_TABLE.json"
    )
    contract_path = (
        output_dir
        / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT.json"
    )

    contract: dict[str, Any] = {}

    if not failures:
        np.save(
            coords_path,
            coordinates,
            allow_pickle=False,
        )
        np.save(
            edge_path,
            edge_index,
            allow_pickle=False,
        )
        np.save(
            adjacency_path,
            adjacency,
            allow_pickle=False,
        )
        write_json(neighbors_path, neighbors)

        contract_core = {
            "name": (
                "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT"
            ),
            "contract_version": 1,
            "source_kind": (
                "canonical_generation_from_frozen_G0_4x4_mesh_contract"
            ),
            "recovered_edge_artifact_found_in_p2_dataset": False,
            "new_explicit_numbering_decision": True,
            "topology": "4x4_2D_MESH",
            "num_nodes": 16,
            "router_ids": list(range(16)),
            "router_numbering": "row_major",
            "router_id_equation": "router_id = row * 4 + column",
            "coordinate_semantics": {
                "coordinate_0": "row_vertical",
                "coordinate_1": "column_horizontal",
            },
            "physical_routing_graph": {
                "directed": True,
                "bidirectional_links": True,
                "physical_self_loops": False,
                "directed_edge_count": 48,
                "undirected_link_count": 24,
                "edge_order": (
                    "lexicographic ascending by (source_router, "
                    "destination_router)"
                ),
            },
            "graph_operator_rules": {
                "GraphConv": (
                    "consume the stored physical edge_index directly"
                ),
                "GATConv": (
                    "operator self-loop behavior must follow the frozen "
                    "G0 operator contract"
                ),
                "GCNConv": (
                    "the stored physical graph has no self-loops; GCNConv "
                    "may add normalized self-loops according to G0"
                ),
            },
            "batching_rule": (
                "for batched graphs, replicate this fixed edge list with "
                "node-index offsets of 16 per graph"
            ),
            "artifacts": {
                "coordinates": {
                    "filename": coords_path.name,
                    "sha256": sha256_file(coords_path),
                    "shape": [16, 2],
                    "dtype": "int64",
                },
                "edge_index": {
                    "filename": edge_path.name,
                    "sha256": sha256_file(edge_path),
                    "shape": [2, 48],
                    "dtype": "int64",
                },
                "adjacency": {
                    "filename": adjacency_path.name,
                    "sha256": sha256_file(adjacency_path),
                    "shape": [16, 16],
                    "dtype": "uint8",
                },
                "neighbor_table": {
                    "filename": neighbors_path.name,
                    "sha256": sha256_file(neighbors_path),
                },
            },
            "provenance": {
                "g0_protocol_sha256": EXPECTED_G0_PROTOCOL_SHA,
                "g0_operator_contract_file_sha256": (
                    sha256_file(g0_operator_path)
                ),
                "r2_hold_report_sha256": (
                    sha256_file(r2_report_path)
                ),
                "loader_sha256": sha256_file(loader),
            },
        }
        contract = {
            **contract_core,
            "contract_sha256": canonical_sha256(contract_core),
        }
        write_json(contract_path, contract)

    report = {
        "stage": STAGE,
        "status": "COMPLETE" if not failures else "HOLD",
        "decision": (
            "FREEZE_CANONICAL_ROW_MAJOR_4X4_TOPOLOGY_AND_AUTHORIZE_G1"
            if not failures
            else "HOLD_CANONICAL_TOPOLOGY_CONTRACT"
        ),
        "interpretation": (
            "The P2 Conv1D dataset does not contain edge_index because "
            "the frozen B3 architecture did not require graph message "
            "passing. G1 therefore requires an explicit static topology "
            "contract rather than recovery of a nonexistent sample field."
        ),
        "historical_g1a_hold_preserved": True,
        "historical_g1a_r1_hold_preserved": True,
        "historical_g1a_r2_hold_preserved": True,
        "source_audit": evidence,
        "canonical_contract": contract,
        "security_boundary": {
            "p2_train_sample_deserialized": False,
            "p2_validation_sample_deserialized": False,
            "p2_test_directory_enumerated": False,
            "p2_test_tensors_deserialized": False,
            "checkpoint_loaded": False,
            "b4_validation_cache_accessed": False,
            "b6_test_cache_accessed": False,
            "training_performed": False,
            "optimization_steps": 0,
            "threshold_tuning_performed": False,
            "architecture_selected": False,
            "quantization_performed": False,
            "rtl_generated": False,
            "legal_decoder_implemented": False,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P2_G1_SOURCE_ONLY_GRAPH_BASELINE_IMPLEMENTATION_AND_TRAINING"
            if not failures
            else "V5_P2_G1A_R2A_R1_CANONICAL_TOPOLOGY_CONTRACT"
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    markdown = [
        "# V5 P2-G1A-R2A Canonical Static Topology Contract",
        "",
        f"- Status: **{report['status']}**",
        "",
        "The P2 Conv1D dataset does not carry `edge_index` because the "
        "frozen B3 model did not use graph message passing. This stage "
        "therefore freezes a new explicit graph input rather than pretending "
        "to recover a missing sample field.",
        "",
        "## Frozen topology",
        "",
        "- 4×4 2D physical mesh.",
        "- Router IDs 0–15.",
        "- Row-major numbering: `router_id = row * 4 + column`.",
        "- 24 physical bidirectional links.",
        "- 48 directed edges.",
        "- No physical self-loops.",
        "- Lexicographically sorted directed edge order.",
        "",
        "The generated topology is now the only authorized edge input for "
        "G1 graph baselines.",
        "",
        "No model training, checkpoint loading, prediction-cache access, "
        "P2 test access, quantization, RTL generation, or Legal NoC decoder "
        "implementation was performed.",
        "",
    ]
    atomic_write(
        output_dir
        / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT.md",
        "\n".join(markdown),
    )

    lock = {
        "status": COMPLETE if not failures else f"{STAGE}_HOLD",
        "report_sha256": sha256_file(report_path),
        "contract_file_sha256": (
            sha256_file(contract_path)
            if contract_path.is_file()
            else None
        ),
        "contract_sha256": (
            contract.get("contract_sha256")
            if contract
            else None
        ),
        "coordinates_sha256": (
            sha256_file(coords_path)
            if coords_path.is_file()
            else None
        ),
        "edge_index_sha256": (
            sha256_file(edge_path)
            if edge_path.is_file()
            else None
        ),
        "adjacency_sha256": (
            sha256_file(adjacency_path)
            if adjacency_path.is_file()
            else None
        ),
        "neighbor_table_sha256": (
            sha256_file(neighbors_path)
            if neighbors_path.is_file()
            else None
        ),
        "historical_holds_preserved": True,
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "checkpoint_loaded": False,
        "b4_validation_cache_accessed": False,
        "b6_test_cache_accessed": False,
        "training_performed": False,
        "optimization_steps": 0,
        "architecture_selected": False,
        "quantization_performed": False,
        "rtl_generated": False,
        "legal_decoder_implemented": False,
        "next_stage": report["next_stage"],
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(
        output_dir
        / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT_LOCK.json",
        lock,
    )

    if failures:
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print("===== V5 P2-G1A-R2A CANONICAL TOPOLOGY CONTRACT =====")
        print("status: HOLD")
        print("historical_holds_preserved: true")
        print("failure_count:", len(failures))
        print("warning_count:", len(warnings))
        for failure in failures:
            print("FAIL:", failure)
        print(f"{STAGE}_HOLD")
        return 1

    atomic_write(
        output_dir / COMPLETE,
        COMPLETE + "\n",
    )

    print("===== V5 P2-G1A-R2A CANONICAL TOPOLOGY CONTRACT =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "FREEZE_CANONICAL_ROW_MAJOR_4X4_TOPOLOGY_AND_AUTHORIZE_G1"
    )
    print("historical_g1a_hold_preserved: true")
    print("historical_g1a_r1_hold_preserved: true")
    print("historical_g1a_r2_hold_preserved: true")
    print("recovered_edge_artifact_found: false")
    print("canonical_topology_generated: true")
    print("router_numbering: ROW_MAJOR")
    print("router_id_equation: router_id=row*4+column")
    print("num_nodes: 16")
    print("undirected_physical_links: 24")
    print("directed_edges: 48")
    print("physical_self_loops: false")
    print("edge_order: LEXICOGRAPHIC_SOURCE_DESTINATION")
    print(
        "edge_index_artifact_sha256:",
        sha256_file(edge_path),
    )
    print(
        "topology_contract_sha256:",
        contract["contract_sha256"],
    )
    print(
        "topology_evidence_files_scanned:",
        evidence.get("files_scanned"),
    )
    print("p2_test_directory_enumerated: false")
    print("p2_test_tensors_deserialized: false")
    print("checkpoint_loaded: false")
    print("b4_validation_cache_accessed: false")
    print("b6_test_cache_accessed: false")
    print("training_performed: false")
    print("optimization_steps: 0")
    print("architecture_selected: false")
    print("failure_count: 0")
    print("warning_count:", len(warnings))
    print(
        "next_stage: "
        "V5_P2_G1_SOURCE_ONLY_GRAPH_BASELINE_IMPLEMENTATION_AND_TRAINING"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
