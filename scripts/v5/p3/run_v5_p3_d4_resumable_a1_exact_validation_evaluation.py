from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGE = "V5_P3_D4_RESUMABLE_A1_EXACT_TRANCHE_A_VALIDATION_EVALUATION"
CAMPAIGN_LABEL = "V5-P3-1500-D70 Tranche-A Preliminary Diagnostic"
EXPECTED_ITEMS = 13_863
ROUTER_COUNT = 16
CHUNK_SIZE = 100
A1_MARGIN_THRESHOLD = 8.7205320882398425

FIELD_ALIASES = {
    "graph": (
        "attack",
        "is_attack",
        "graph",
        "graph_prediction",
        "attack_prediction",
        "predicted_attack",
        "y_attack",
    ),
    "count": (
        "attacker_count",
        "count",
        "predicted_count",
        "source_count",
        "k",
    ),
    "source": (
        "source_mask",
        "sources_mask",
        "source_bitmap",
        "source",
        "sources",
    ),
    "transit": (
        "transit_mask",
        "transits_mask",
        "transit_bitmap",
        "transit",
        "transits",
    ),
    "victim": (
        "victim_mask",
        "victims_mask",
        "victim_bitmap",
        "victim",
        "victims",
    ),
    "path": (
        "path_mask",
        "attack_path_mask",
        "path_bitmap",
        "attack_path",
        "path",
    ),
    "route_ids": (
        "route_ids",
        "routes",
        "selected_route_ids",
    ),
    "margin": (
        "margin",
        "attack_margin",
        "graph_margin",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--installed-script", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(
                f"non-finite value in decoder object: {value}"
            )
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            key: jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def normalized_key(value: str) -> str:
    return "".join(
        character.lower()
        for character in value
        if character.isalnum() or character == "_"
    )


def iter_mapping_nodes(
    value: Any,
    path: str = "$",
) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from iter_mapping_nodes(
                item,
                f"{path}.{key}",
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_mapping_nodes(
                item,
                f"{path}[{index}]",
            )


def resolve_field(
    document: dict[str, Any],
    semantic_name: str,
) -> tuple[Any, str]:
    aliases = {
        normalized_key(alias)
        for alias in FIELD_ALIASES[semantic_name]
    }

    # Prefer the top-level decoded-output contract.
    for key, value in document.items():
        if normalized_key(str(key)) in aliases:
            return value, f"$.{key}"

    candidates: list[tuple[int, str, Any]] = []
    for path, mapping in iter_mapping_nodes(document):
        depth = path.count(".") + path.count("[")
        for key, value in mapping.items():
            if normalized_key(str(key)) in aliases:
                candidates.append(
                    (depth, f"{path}.{key}", value)
                )
    if not candidates:
        raise KeyError(
            f"cannot resolve {semantic_name}; "
            f"top-level keys={sorted(document)}"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, path, value = candidates[0]
    return value, path


def scalar_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite graph scalar")
        if value in (0.0, 1.0):
            return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "attack", "active", "1"}:
            return True
        if lowered in {"false", "normal", "inactive", "0"}:
            return False
    raise ValueError(f"cannot convert graph field to bool: {value!r}")


def scalar_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value) and float(value).is_integer():
            return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"cannot convert count field to int: {value!r}")


def mask_vector(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        # Common serialized forms: {"mask": ...} or
        # {"bits": [...]}.
        for key in ("mask", "bitmap", "bits", "value"):
            if key in value:
                return mask_vector(value[key])

    if isinstance(value, (int, np.integer)):
        integer = int(value)
        if integer < 0 or integer >= (1 << ROUTER_COUNT):
            raise ValueError(
                f"integer mask out of range: {integer}"
            )
        return np.asarray(
            [
                (integer >> router) & 1
                for router in range(ROUTER_COUNT)
            ],
            dtype=np.uint8,
        )

    array = np.asarray(value)
    if array.ndim == 0:
        return mask_vector(array.item())
    flat = array.reshape(-1)
    if flat.size == ROUTER_COUNT:
        if not np.all(np.isin(flat, [0, 1, False, True])):
            raise ValueError(
                f"16-element mask is not binary: {flat.tolist()}"
            )
        return flat.astype(np.uint8)

    # A short list of router IDs is also accepted.
    if flat.size <= ROUTER_COUNT:
        try:
            indices = [int(item) for item in flat.tolist()]
        except Exception as exc:
            raise ValueError(
                f"cannot interpret mask value: {value!r}"
            ) from exc
        if len(indices) != len(set(indices)):
            raise ValueError(
                f"duplicate router IDs in mask: {indices}"
            )
        if all(0 <= item < ROUTER_COUNT for item in indices):
            result = np.zeros(ROUTER_COUNT, dtype=np.uint8)
            result[indices] = 1
            return result

    raise ValueError(
        f"cannot convert mask to {ROUTER_COUNT} bits: {value!r}"
    )


def route_ids_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.int32)
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = int(array.item())
        return np.asarray([scalar], dtype=np.int32)
    return np.asarray(
        [int(item) for item in array.reshape(-1).tolist()],
        dtype=np.int32,
    )


def extract_decoded_output(
    decoded: Any,
    hypothesis: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    decoded_document = jsonable(decoded)
    hypothesis_document = jsonable(hypothesis)
    if not isinstance(decoded_document, dict):
        raise RuntimeError(
            "decoded output is not mapping-like: "
            f"{type(decoded_document)}"
        )
    if not isinstance(hypothesis_document, dict):
        raise RuntimeError(
            "hypothesis is not mapping-like: "
            f"{type(hypothesis_document)}"
        )

    paths: dict[str, str] = {}

    graph_value, paths["graph"] = resolve_field(
        decoded_document,
        "graph",
    )
    graph = scalar_bool(graph_value)

    masks = {}
    for role in ("source", "transit", "victim", "path"):
        try:
            value, paths[role] = resolve_field(
                decoded_document,
                role,
            )
        except KeyError:
            if not graph:
                masks[role] = np.zeros(
                    ROUTER_COUNT,
                    dtype=np.uint8,
                )
                paths[role] = (
                    "$.<implicit_zero_for_normal_output>"
                )
                continue
            value, paths[role] = resolve_field(
                hypothesis_document,
                role,
            )
        masks[role] = mask_vector(value)

    try:
        count_value, paths["count"] = resolve_field(
            decoded_document,
            "count",
        )
        count = scalar_int(count_value)
    except KeyError:
        if graph:
            count = int(masks["source"].sum())
            paths["count"] = "$.<derived_from_source_mask>"
        else:
            count = 0
            paths["count"] = "$.<implicit_zero_for_normal_output>"

    try:
        route_value, paths["route_ids"] = resolve_field(
            decoded_document,
            "route_ids",
        )
    except KeyError:
        try:
            route_value, paths["route_ids"] = resolve_field(
                hypothesis_document,
                "route_ids",
            )
        except KeyError:
            route_value = []
            paths["route_ids"] = "$.<not_exposed>"
    route_ids = route_ids_vector(route_value)

    try:
        margin_value, paths["margin"] = resolve_field(
            hypothesis_document,
            "margin",
        )
        margin = float(margin_value)
    except KeyError:
        margin = float(getattr(hypothesis, "margin"))
        paths["margin"] = "$.<hypothesis_attribute>"

    if not math.isfinite(margin):
        raise RuntimeError("decoder produced non-finite margin")

    if not graph:
        if count != 0:
            raise RuntimeError(
                f"normal decoded output has count={count}"
            )
        if any(int(mask.sum()) != 0 for mask in masks.values()):
            raise RuntimeError(
                "normal decoded output has non-empty role masks"
            )
        route_ids = np.empty(0, dtype=np.int32)
    else:
        if count not in (1, 2, 3, 4):
            raise RuntimeError(
                f"attack decoded output has invalid count={count}"
            )

    return {
        "graph": np.uint8(graph),
        "count": np.int8(count),
        "source": masks["source"],
        "transit": masks["transit"],
        "victim": masks["victim"],
        "path": masks["path"],
        "route_ids": route_ids,
        "margin": np.float64(margin),
        "decoded_document": decoded_document,
        "hypothesis_document": hypothesis_document,
    }, paths


def structural_legality(
    graph: int,
    count: int,
    source: np.ndarray,
    transit: np.ndarray,
    victim: np.ndarray,
    path: np.ndarray,
    route_ids: np.ndarray,
) -> dict[str, bool]:
    if graph == 0:
        normal_empty = (
            count == 0
            and source.sum() == 0
            and transit.sum() == 0
            and victim.sum() == 0
            and path.sum() == 0
            and route_ids.size == 0
        )
        return {
            "normal_output_empty": bool(normal_empty),
            "source_count_matches_count": bool(normal_empty),
            "has_at_least_one_victim": bool(normal_empty),
            "endpoints_inside_path": bool(normal_empty),
            "transit_inside_path": bool(normal_empty),
            "path_equals_role_union": bool(normal_empty),
            "route_count_matches_count": bool(normal_empty),
            "source_victim_disjoint_observed": bool(normal_empty),
            "transit_endpoint_disjoint_observed": bool(normal_empty),
            "role_overlap_is_permitted_under_union_semantics": True,
            "all_structural_checks": bool(normal_empty),
        }

    source_bool = source.astype(bool)
    transit_bool = transit.astype(bool)
    victim_bool = victim.astype(bool)
    path_bool = path.astype(bool)

    # The role masks are UNIONS across all selected routes. A router may be
    # an endpoint on one route and a transit router on another route, so
    # pairwise union-mask disjointness is diagnostic rather than a legality
    # invariant. Route-level legality is already certified inside the frozen
    # exact decoder.
    required_checks = {
        "normal_output_empty": True,
        "source_count_matches_count": (
            int(source.sum()) == count
        ),
        "has_at_least_one_victim": (
            int(victim.sum()) >= 1
        ),
        "endpoints_inside_path": bool(
            np.all(
                ~(source_bool | victim_bool)
                | path_bool
            )
        ),
        "transit_inside_path": bool(
            np.all(~transit_bool | path_bool)
        ),
        "path_equals_role_union": bool(
            np.all(
                path_bool
                == (
                    source_bool
                    | transit_bool
                    | victim_bool
                )
            )
        ),
        "route_count_matches_count": (
            route_ids.size in (0, count)
        ),
    }
    checks = {
        **required_checks,
        "source_victim_disjoint_observed": bool(
            not np.any(source_bool & victim_bool)
        ),
        "transit_endpoint_disjoint_observed": bool(
            not np.any(
                transit_bool & (source_bool | victim_bool)
            )
        ),
        "role_overlap_is_permitted_under_union_semantics": True,
    }
    checks["all_structural_checks"] = all(
        required_checks.values()
    )
    return checks


def chunk_ranges(total: int):
    for chunk_id, start in enumerate(
        range(0, total, CHUNK_SIZE)
    ):
        stop = min(start + CHUNK_SIZE, total)
        yield chunk_id, start, stop


def chunk_paths(
    working_dir: Path,
    chunk_id: int,
    start: int,
    stop: int,
) -> tuple[Path, Path]:
    stem = (
        f"chunk_{chunk_id:04d}_"
        f"items_{start:05d}_{stop:05d}"
    )
    return (
        working_dir / f"{stem}.npz",
        working_dir / f"{stem}.manifest.json",
    )


def validate_chunk(
    chunk_path: Path,
    manifest_path: Path,
    expected_start: int,
    expected_stop: int,
    decoder_sha256: str,
    export_sha256: str,
) -> dict[str, np.ndarray]:
    if not chunk_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError(
            f"chunk manifest not COMPLETE: {manifest_path}"
        )
    if int(manifest.get("start", -1)) != expected_start:
        raise RuntimeError(
            f"chunk start changed: {manifest_path}"
        )
    if int(manifest.get("stop", -1)) != expected_stop:
        raise RuntimeError(
            f"chunk stop changed: {manifest_path}"
        )
    if manifest.get(
        "certified_decoder_sha256"
    ) != decoder_sha256:
        raise RuntimeError(
            f"decoder hash changed for {manifest_path}"
        )
    if manifest.get(
        "D1_export_sha256"
    ) != export_sha256:
        raise RuntimeError(
            f"D1 export hash changed for {manifest_path}"
        )
    if float(
        manifest.get("A1_margin_threshold")
    ) != A1_MARGIN_THRESHOLD:
        raise RuntimeError(
            f"A1 margin changed for {manifest_path}"
        )
    if manifest.get("chunk_sha256") != sha256_file(
        chunk_path
    ):
        raise RuntimeError(
            f"chunk SHA mismatch: {chunk_path}"
        )

    with np.load(
        chunk_path,
        allow_pickle=False,
    ) as loaded:
        arrays = {
            key: loaded[key].copy()
            for key in loaded.files
        }

    expected_items = expected_stop - expected_start
    expected_shapes = {
        "dataset_index": (expected_items,),
        "margin": (expected_items,),
        "graph_prediction": (expected_items,),
        "count_prediction": (expected_items,),
        "source_prediction": (
            expected_items,
            ROUTER_COUNT,
        ),
        "transit_prediction": (
            expected_items,
            ROUTER_COUNT,
        ),
        "victim_prediction": (
            expected_items,
            ROUTER_COUNT,
        ),
        "path_prediction": (
            expected_items,
            ROUTER_COUNT,
        ),
        "route_count": (expected_items,),
        "decode_seconds": (expected_items,),
        "structural_legal": (expected_items,),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in arrays:
            raise RuntimeError(
                f"chunk missing {key}: {chunk_path}"
            )
        if arrays[key].shape != expected_shape:
            raise RuntimeError(
                f"{key} shape={arrays[key].shape}, "
                f"expected={expected_shape}"
            )

    expected_indices = np.arange(
        expected_start,
        expected_stop,
        dtype=np.int32,
    )
    if not np.array_equal(
        arrays["dataset_index"],
        expected_indices,
    ):
        raise RuntimeError(
            f"chunk index coverage changed: {chunk_path}"
        )
    if not np.isfinite(arrays["margin"]).all():
        raise RuntimeError(
            f"non-finite margin in {chunk_path}"
        )
    if not np.isfinite(
        arrays["decode_seconds"]
    ).all():
        raise RuntimeError(
            f"non-finite timing in {chunk_path}"
        )
    if not np.all(
        arrays["structural_legal"] == 1
    ):
        raise RuntimeError(
            f"structural legality failure in {chunk_path}"
        )
    return arrays


def decode_chunk(
    decoder,
    arrays: dict[str, np.ndarray],
    start: int,
    stop: int,
    schema_artifact_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    item_count = stop - start
    output = {
        "dataset_index": np.arange(
            start,
            stop,
            dtype=np.int32,
        ),
        "margin": np.empty(
            item_count,
            dtype=np.float64,
        ),
        "graph_prediction": np.empty(
            item_count,
            dtype=np.uint8,
        ),
        "count_prediction": np.empty(
            item_count,
            dtype=np.int8,
        ),
        "source_prediction": np.empty(
            (item_count, ROUTER_COUNT),
            dtype=np.uint8,
        ),
        "transit_prediction": np.empty(
            (item_count, ROUTER_COUNT),
            dtype=np.uint8,
        ),
        "victim_prediction": np.empty(
            (item_count, ROUTER_COUNT),
            dtype=np.uint8,
        ),
        "path_prediction": np.empty(
            (item_count, ROUTER_COUNT),
            dtype=np.uint8,
        ),
        "route_count": np.empty(
            item_count,
            dtype=np.int8,
        ),
        "decode_seconds": np.empty(
            item_count,
            dtype=np.float64,
        ),
        "structural_legal": np.empty(
            item_count,
            dtype=np.uint8,
        ),
    }
    schema_value = None

    for local, dataset_index in enumerate(
        range(start, stop)
    ):
        started = time.perf_counter()
        hypothesis = (
            decoder.decode_best_attack_hypothesis(
                float(
                    arrays["attack_logits"][
                        dataset_index
                    ]
                ),
                arrays["count_logits"][
                    dataset_index
                ].astype(np.float64),
                arrays["source_logits"][
                    dataset_index
                ].astype(np.float64),
                arrays["transit_logits"][
                    dataset_index
                ].astype(np.float64),
                arrays["victim_logits"][
                    dataset_index
                ].astype(np.float64),
                arrays["path_logits"][
                    dataset_index
                ].astype(np.float64),
            )
        )
        decoded = decoder.apply_margin_threshold(
            hypothesis,
            A1_MARGIN_THRESHOLD,
        )
        elapsed = time.perf_counter() - started

        extracted, paths = extract_decoded_output(
            decoded,
            hypothesis,
        )
        legality = structural_legality(
            int(extracted["graph"]),
            int(extracted["count"]),
            extracted["source"],
            extracted["transit"],
            extracted["victim"],
            extracted["path"],
            extracted["route_ids"],
        )
        if not legality["all_structural_checks"]:
            raise RuntimeError(
                "A1 structural legality failure at "
                f"dataset_index={dataset_index}: "
                f"{legality}"
            )

        output["margin"][local] = (
            extracted["margin"]
        )
        output["graph_prediction"][local] = (
            extracted["graph"]
        )
        output["count_prediction"][local] = (
            extracted["count"]
        )
        for role in (
            "source",
            "transit",
            "victim",
            "path",
        ):
            output[f"{role}_prediction"][
                local
            ] = extracted[role]
        output["route_count"][local] = int(
            extracted["route_ids"].size
        )
        output["decode_seconds"][local] = elapsed
        output["structural_legal"][local] = 1

        if schema_value is None:
            schema_value = {
                "stage": STAGE,
                "dataset_index": dataset_index,
                "resolved_paths": paths,
                "decoded_top_level_keys": sorted(
                    extracted[
                        "decoded_document"
                    ]
                ),
                "hypothesis_top_level_keys": sorted(
                    extracted[
                        "hypothesis_document"
                    ]
                ),
                "decoded_example": extracted[
                    "decoded_document"
                ],
                "hypothesis_example": extracted[
                    "hypothesis_document"
                ],
            }
            if not schema_artifact_path.is_file():
                atomic_json(
                    schema_artifact_path,
                    schema_value,
                )

        if (
            local == 0
            or (local + 1) % 20 == 0
            or local + 1 == item_count
        ):
            print(
                f"chunk_item={local + 1}/"
                f"{item_count} "
                f"dataset_index={dataset_index} "
                f"seconds={elapsed:.6f}",
                flush=True,
            )

    if schema_value is None:
        raise RuntimeError("empty chunk")
    return output, schema_value


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(
        prediction,
        dtype=np.uint8,
    ).reshape(-1)
    target = np.asarray(
        target,
        dtype=np.uint8,
    ).reshape(-1)
    tp = int(
        np.sum(
            (prediction == 1) & (target == 1)
        )
    )
    fp = int(
        np.sum(
            (prediction == 1) & (target == 0)
        )
    )
    fn = int(
        np.sum(
            (prediction == 0) & (target == 1)
        )
    )
    tn = int(
        np.sum(
            (prediction == 0) & (target == 0)
        )
    )
    precision = (
        tp / (tp + fp) if tp + fp else 0.0
    )
    recall = (
        tp / (tp + fn) if tp + fn else 0.0
    )
    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "accuracy": (
            (tp + tn)
            / max(1, tp + fp + fn + tn)
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": (
            fp / (fp + tn)
            if fp + tn
            else 0.0
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(
        values,
        kind="mergesort",
    )
    sorted_values = values[order]
    ranks = np.empty(
        len(values),
        dtype=np.float64,
    )
    start = 0
    while start < len(values):
        stop = start + 1
        while (
            stop < len(values)
            and sorted_values[stop]
            == sorted_values[start]
        ):
            stop += 1
        ranks[order[start:stop]] = (
            start + 1 + stop
        ) / 2.0
        start = stop
    return ranks


def binary_auroc(
    scores: np.ndarray,
    targets: np.ndarray,
) -> float:
    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)
    targets = np.asarray(
        targets,
        dtype=np.int64,
    ).reshape(-1)
    positive = targets == 1
    negative = targets == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = average_ranks(scores)
    return float(
        (
            ranks[positive].sum()
            - n_positive
            * (n_positive + 1)
            / 2.0
        )
        / (n_positive * n_negative)
    )


def average_precision(
    scores: np.ndarray,
    targets: np.ndarray,
) -> float:
    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)
    targets = np.asarray(
        targets,
        dtype=np.int64,
    ).reshape(-1)
    positive_count = int(
        (targets == 1).sum()
    )
    if positive_count == 0:
        return float("nan")
    order = np.argsort(
        -scores,
        kind="mergesort",
    )
    ordered = targets[order]
    cumulative = np.cumsum(ordered == 1)
    positions = np.arange(
        1,
        len(targets) + 1,
    )
    precision = cumulative / positions
    return float(
        precision[ordered == 1].sum()
        / positive_count
    )


def count_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    active_truth: np.ndarray,
) -> dict[str, Any]:
    prediction = prediction[active_truth].astype(
        np.int64
    )
    target = target[active_truth].astype(
        np.int64
    )
    per_class = {}
    f1_values = []
    for class_value in (1, 2, 3, 4):
        pred_positive = (
            prediction == class_value
        )
        true_positive = (
            target == class_value
        )
        tp = int(
            np.sum(
                pred_positive & true_positive
            )
        )
        fp = int(
            np.sum(
                pred_positive & ~true_positive
            )
        )
        fn = int(
            np.sum(
                ~pred_positive & true_positive
            )
        )
        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )
        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )
        f1 = (
            2.0 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[str(class_value)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(
                true_positive.sum()
            ),
        }
    return {
        "accuracy": float(
            np.mean(prediction == target)
        ),
        "macro_f1": float(
            np.mean(f1_values)
        ),
        "predicted_zero_rate": float(
            np.mean(prediction == 0)
        ),
        "items": int(target.size),
        "per_class": per_class,
    }


def role_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    active_truth: np.ndarray,
) -> dict[str, Any]:
    flattened = binary_metrics(
        prediction.reshape(-1),
        target.reshape(-1),
    )
    return {
        "flattened": flattened,
        "exact_all": float(
            np.mean(
                np.all(
                    prediction == target,
                    axis=1,
                )
            )
        ),
        "exact_active": float(
            np.mean(
                np.all(
                    prediction[active_truth]
                    == target[active_truth],
                    axis=1,
                )
            )
        ),
        "exact_inactive": float(
            np.mean(
                np.all(
                    prediction[~active_truth]
                    == target[~active_truth],
                    axis=1,
                )
            )
        ),
        "mean_predicted_cardinality_active": float(
            prediction[
                active_truth
            ].sum(axis=1).mean()
        ),
        "mean_true_cardinality_active": float(
            target[
                active_truth
            ].sum(axis=1).mean()
        ),
    }


def strict_metrics(
    graph_prediction: np.ndarray,
    count_prediction: np.ndarray,
    role_predictions: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    graph_truth = labels["graph"]
    active_truth = graph_truth == 1
    graph_correct = (
        graph_prediction == graph_truth
    )
    role_correct = np.ones(
        graph_truth.shape[0],
        dtype=bool,
    )
    roles_empty = np.ones(
        graph_truth.shape[0],
        dtype=bool,
    )
    for role in (
        "source",
        "transit",
        "victim",
        "path",
    ):
        role_correct &= np.all(
            role_predictions[role]
            == labels[role],
            axis=1,
        )
        roles_empty &= np.all(
            role_predictions[role] == 0,
            axis=1,
        )

    active_exact = (
        active_truth
        & graph_correct
        & (
            count_prediction
            == labels["count"]
        )
        & role_correct
    )
    inactive_exact = (
        (~active_truth)
        & graph_correct
        & roles_empty
    )
    strict = active_exact | inactive_exact
    return {
        "strict_all_task_exactness": float(
            strict.mean()
        ),
        "strict_active_exactness": float(
            active_exact[
                active_truth
            ].mean()
        ),
        "strict_inactive_exactness": float(
            inactive_exact[
                ~active_truth
            ].mean()
        ),
        "strict_exact_items": int(
            strict.sum()
        ),
        "strict_active_exact_items": int(
            active_exact.sum()
        ),
        "strict_inactive_exact_items": int(
            inactive_exact.sum()
        ),
    }


def main() -> int:
    args = parse_args()

    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()
    installed_script = Path(
        args.installed_script
    ).expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    working_dir = output_dir / "A1_WORKING"
    working_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    contract_path = output_dir / (
        "V5_P3_D4_A1_RESUMABLE_EXECUTION_CONTRACT.json"
    )
    schema_path = output_dir / (
        "V5_P3_D4_A1_DECODED_OUTPUT_SCHEMA.json"
    )
    final_outputs_path = output_dir / (
        "V5_P3_D4_A1_EXACT_VALIDATION_OUTPUTS.npz"
    )
    report_path = output_dir / f"{STAGE}_REPORT.json"
    lock_path = output_dir / f"{STAGE}_LOCK.json"
    complete_path = output_dir / f"{STAGE}_COMPLETE"
    failure_path = output_dir / f"{STAGE}_FAILURE.json"

    d3_dir = (
        repo
        / "reports/v5/p3_d3_resumable_a1_exact_decoder_chunk_preflight"
    )
    d3_report_path = d3_dir / (
        "V5_P3_D3_RESUMABLE_A1_EXACT_"
        "DECODER_CHUNK_PREFLIGHT_REPORT.json"
    )
    d3_lock_path = d3_dir / (
        "V5_P3_D3_RESUMABLE_A1_EXACT_"
        "DECODER_CHUNK_PREFLIGHT_LOCK.json"
    )
    d2_dir = (
        repo
        / "reports/v5/p3_d2_raw_and_frozen_a0_validation_evaluation"
    )
    d2_report_path = d2_dir / (
        "V5_P3_D2_RAW_AND_FROZEN_A0_"
        "TRANCHE_A_VALIDATION_EVALUATION_REPORT.json"
    )
    d2_lock_path = d2_dir / (
        "V5_P3_D2_RAW_AND_FROZEN_A0_"
        "TRANCHE_A_VALIDATION_EVALUATION_LOCK.json"
    )
    d1_dir = (
        repo
        / "reports/v5/p3_d1_immutable_validation_logit_export"
    )
    d1_export_path = d1_dir / (
        "V5_P3_D1_TRANCHE_A_VALIDATION_LOGITS_AND_LABELS.npz"
    )
    d1_lock_path = d1_dir / (
        "V5_P3_D1_IMMUTABLE_TRANCHE_A_"
        "VALIDATION_LOGIT_EXPORT_LOCK.json"
    )
    decoder_path = (
        repo
        / "src/decoders/v5_legal_xy_exact_decoder_certified.py"
    )
    preserved_decoder_path = (
        repo
        / "reports/v5/p2_canonical_preservation/decoder/"
        "v5_legal_xy_exact_decoder_certified.py"
    )

    required = [
        d3_report_path,
        d3_lock_path,
        d2_report_path,
        d2_lock_path,
        d1_export_path,
        d1_lock_path,
        decoder_path,
        preserved_decoder_path,
    ]
    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"required artifacts missing: {missing}"
        )

    d3_report = json.loads(
        d3_report_path.read_text(
            encoding="utf-8"
        )
    )
    d3_lock = json.loads(
        d3_lock_path.read_text(
            encoding="utf-8"
        )
    )
    d2_report = json.loads(
        d2_report_path.read_text(
            encoding="utf-8"
        )
    )
    d2_lock = json.loads(
        d2_lock_path.read_text(
            encoding="utf-8"
        )
    )
    d1_lock = json.loads(
        d1_lock_path.read_text(
            encoding="utf-8"
        )
    )

    if d3_report.get("status") != "PASS":
        raise RuntimeError("D3 report is not PASS")
    if not d3_report.get(
        "decision",
        {},
    ).get(
        "A1_exact_full_validation_evaluation_authorized"
    ):
        raise RuntimeError(
            "D3 did not authorize full A1 validation"
        )
    if d3_lock.get(
        "report_sha256"
    ) != sha256_file(d3_report_path):
        raise RuntimeError(
            "D3 report/lock SHA mismatch"
        )
    if d2_lock.get(
        "report_sha256"
    ) != sha256_file(d2_report_path):
        raise RuntimeError(
            "D2 report/lock SHA mismatch"
        )
    export_sha = sha256_file(
        d1_export_path
    )
    if d1_lock.get(
        "export_sha256"
    ) != export_sha:
        raise RuntimeError(
            "D1 export SHA mismatch"
        )

    decoder_sha = sha256_file(
        decoder_path
    )
    preserved_sha = sha256_file(
        preserved_decoder_path
    )
    if decoder_sha != preserved_sha:
        raise RuntimeError(
            "working decoder differs from "
            "canonical preservation copy"
        )
    if d3_lock.get(
        "certified_decoder_sha256"
    ) != decoder_sha:
        raise RuntimeError(
            "decoder differs from D3"
        )
    if float(
        d3_lock.get("A1_margin_threshold")
    ) != A1_MARGIN_THRESHOLD:
        raise RuntimeError(
            "A1 margin differs from D3"
        )

    total_chunks = math.ceil(
        EXPECTED_ITEMS / CHUNK_SIZE
    )
    contract = {
        "stage": STAGE,
        "campaign_label": CAMPAIGN_LABEL,
        "items": EXPECTED_ITEMS,
        "chunk_size": CHUNK_SIZE,
        "chunk_count": total_chunks,
        "row_order": (
            "D1_validation_dataset_index_ascending"
        ),
        "D1_export_path": str(
            d1_export_path
        ),
        "D1_export_sha256": export_sha,
        "certified_decoder_path": str(
            decoder_path
        ),
        "certified_decoder_sha256": (
            decoder_sha
        ),
        "A1_margin_threshold": (
            A1_MARGIN_THRESHOLD
        ),
        "threshold_tuning_performed": False,
        "beam_decoder_selected": False,
        "test_accessed": False,
    }
    if contract_path.is_file():
        existing_contract = json.loads(
            contract_path.read_text(
                encoding="utf-8"
            )
        )
        if existing_contract != contract:
            raise RuntimeError(
                "D4 execution contract changed"
            )
    else:
        atomic_json(
            contract_path,
            contract,
        )

    with np.load(
        d1_export_path,
        allow_pickle=False,
    ) as loaded:
        source_arrays = {
            key: loaded[key].copy()
            for key in loaded.files
        }

    required_shapes = {
        "attack_logits": (EXPECTED_ITEMS,),
        "count_logits": (EXPECTED_ITEMS, 4),
        "source_logits": (
            EXPECTED_ITEMS,
            ROUTER_COUNT,
        ),
        "transit_logits": (
            EXPECTED_ITEMS,
            ROUTER_COUNT,
        ),
        "victim_logits": (
            EXPECTED_ITEMS,
            ROUTER_COUNT,
        ),
        "path_logits": (
            EXPECTED_ITEMS,
            ROUTER_COUNT,
        ),
        "y_attack": (EXPECTED_ITEMS,),
        "y_attacker_count": (
            EXPECTED_ITEMS,
        ),
        "y_source": (
            EXPECTED_ITEMS,
            ROUTER_COUNT,
        ),
        "y_transit": (
            EXPECTED_ITEMS,
            ROUTER_COUNT,
        ),
        "y_victim": (
            EXPECTED_ITEMS,
            ROUTER_COUNT,
        ),
        "y_attack_path": (
            EXPECTED_ITEMS,
            ROUTER_COUNT,
        ),
    }
    for key, expected_shape in required_shapes.items():
        if key not in source_arrays:
            raise RuntimeError(
                f"D1 export missing {key}"
            )
        if source_arrays[key].shape != expected_shape:
            raise RuntimeError(
                f"{key} shape="
                f"{source_arrays[key].shape}, "
                f"expected={expected_shape}"
            )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    decoder = importlib.import_module(
        "src.decoders."
        "v5_legal_xy_exact_decoder_certified"
    )

    completed_chunks = 0
    reused_chunks = 0
    all_chunks: list[
        dict[str, np.ndarray]
    ] = []

    for chunk_id, start, stop in chunk_ranges(
        EXPECTED_ITEMS
    ):
        chunk_path, manifest_path = chunk_paths(
            working_dir,
            chunk_id,
            start,
            stop,
        )
        try:
            chunk = validate_chunk(
                chunk_path,
                manifest_path,
                start,
                stop,
                decoder_sha,
                export_sha,
            )
        except FileNotFoundError:
            print(
                f"D4_chunk={chunk_id + 1}/"
                f"{total_chunks} "
                f"status=DECODING "
                f"items={start}:{stop}",
                flush=True,
            )
            try:
                chunk, schema_value = decode_chunk(
                    decoder,
                    source_arrays,
                    start,
                    stop,
                    schema_path,
                )
            except Exception as exc:
                atomic_json(
                    failure_path,
                    {
                        "stage": STAGE,
                        "status": "INTERRUPTED_OR_FAILED",
                        "created_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "chunk_id": chunk_id,
                        "start": start,
                        "stop": stop,
                        "exception_type": type(
                            exc
                        ).__name__,
                        "exception": repr(exc),
                        "completed_chunks_before_failure": (
                            completed_chunks
                            + reused_chunks
                        ),
                        "resume_policy": (
                            "rerun exact same launcher; "
                            "completed hash-valid chunks "
                            "will be reused"
                        ),
                        "test_accessed": False,
                    },
                )
                raise

            atomic_npz(
                chunk_path,
                chunk,
            )
            manifest = {
                "stage": STAGE,
                "status": "COMPLETE",
                "chunk_id": chunk_id,
                "start": start,
                "stop": stop,
                "item_count": stop - start,
                "chunk_path": str(
                    chunk_path
                ),
                "chunk_sha256": sha256_file(
                    chunk_path
                ),
                "D1_export_sha256": export_sha,
                "certified_decoder_sha256": (
                    decoder_sha
                ),
                "A1_margin_threshold": (
                    A1_MARGIN_THRESHOLD
                ),
                "all_structural_legal": True,
                "test_accessed": False,
            }
            atomic_json(
                manifest_path,
                manifest,
            )
            chunk = validate_chunk(
                chunk_path,
                manifest_path,
                start,
                stop,
                decoder_sha,
                export_sha,
            )
            completed_chunks += 1
            print(
                f"D4_chunk={chunk_id + 1}/"
                f"{total_chunks} "
                f"status=COMMITTED "
                f"sha256={sha256_file(chunk_path)}",
                flush=True,
            )
        else:
            reused_chunks += 1
            print(
                f"D4_chunk={chunk_id + 1}/"
                f"{total_chunks} "
                f"status=REUSED",
                flush=True,
            )
        all_chunks.append(chunk)

    merged = {
        key: np.concatenate(
            [chunk[key] for chunk in all_chunks],
            axis=0,
        )
        for key in all_chunks[0]
    }
    if merged["dataset_index"].shape != (
        EXPECTED_ITEMS,
    ):
        raise RuntimeError(
            "merged D4 item count mismatch"
        )
    if not np.array_equal(
        merged["dataset_index"],
        np.arange(
            EXPECTED_ITEMS,
            dtype=np.int32,
        ),
    ):
        raise RuntimeError(
            "merged D4 coverage is not contiguous"
        )
    if not np.all(
        merged["structural_legal"] == 1
    ):
        raise RuntimeError(
            "merged D4 legality failure"
        )

    atomic_npz(
        final_outputs_path,
        merged,
    )

    labels = {
        "graph": source_arrays[
            "y_attack"
        ].astype(np.uint8),
        "count": source_arrays[
            "y_attacker_count"
        ].astype(np.int8),
        "source": source_arrays[
            "y_source"
        ].astype(np.uint8),
        "transit": source_arrays[
            "y_transit"
        ].astype(np.uint8),
        "victim": source_arrays[
            "y_victim"
        ].astype(np.uint8),
        "path": source_arrays[
            "y_attack_path"
        ].astype(np.uint8),
    }
    role_predictions = {
        role: merged[
            f"{role}_prediction"
        ]
        for role in (
            "source",
            "transit",
            "victim",
            "path",
        )
    }
    active_truth = (
        labels["graph"] == 1
    )

    graph_metrics = binary_metrics(
        merged["graph_prediction"],
        labels["graph"],
    )
    graph_metrics[
        "margin_auroc"
    ] = binary_auroc(
        merged["margin"],
        labels["graph"],
    )
    graph_metrics[
        "margin_average_precision"
    ] = average_precision(
        merged["margin"],
        labels["graph"],
    )

    metrics = {
        "graph": graph_metrics,
        "count_active": count_metrics(
            merged["count_prediction"],
            labels["count"],
            active_truth,
        ),
        "roles": {
            role: role_metrics(
                role_predictions[role],
                labels[role],
                active_truth,
            )
            for role in (
                "source",
                "transit",
                "victim",
                "path",
            )
        },
        "strict": strict_metrics(
            merged["graph_prediction"],
            merged["count_prediction"],
            role_predictions,
            labels,
        ),
        "legality": {
            "audit_semantics": (
                "route legality certified by exact decoder; evaluator "
                "checks union-valid structural invariants"
            ),
            "source_victim_overlap_items": int(
                np.sum(
                    np.any(
                        role_predictions["source"].astype(bool)
                        & role_predictions["victim"].astype(bool),
                        axis=1,
                    )
                )
            ),
            "transit_endpoint_overlap_items": int(
                np.sum(
                    np.any(
                        role_predictions["transit"].astype(bool)
                        & (
                            role_predictions["source"].astype(bool)
                            | role_predictions["victim"].astype(bool)
                        ),
                        axis=1,
                    )
                )
            ),
            "certified_decoder_items": (
                EXPECTED_ITEMS
            ),
            "certification_exceptions": 0,
            "structural_legal_items": int(
                merged[
                    "structural_legal"
                ].sum()
            ),
            "structural_legality_rate": float(
                merged[
                    "structural_legal"
                ].mean()
            ),
            "route_legal_by_certified_decoder": (
                True
            ),
        },
        "timing": {
            "mean_seconds": float(
                merged[
                    "decode_seconds"
                ].mean()
            ),
            "median_seconds": float(
                np.median(
                    merged[
                        "decode_seconds"
                    ]
                )
            ),
            "p95_seconds": float(
                np.quantile(
                    merged[
                        "decode_seconds"
                    ],
                    0.95,
                )
            ),
            "p99_seconds": float(
                np.quantile(
                    merged[
                        "decode_seconds"
                    ],
                    0.99,
                )
            ),
            "maximum_seconds": float(
                merged[
                    "decode_seconds"
                ].max()
            ),
            "total_recorded_decode_seconds": float(
                merged[
                    "decode_seconds"
                ].sum()
            ),
        },
        "margin": {
            "minimum": float(
                merged["margin"].min()
            ),
            "median": float(
                np.median(
                    merged["margin"]
                )
            ),
            "maximum": float(
                merged["margin"].max()
            ),
            "finite": bool(
                np.isfinite(
                    merged["margin"]
                ).all()
            ),
        },
    }

    raw = d2_report["raw"]
    a0 = d2_report[
        "A0_frozen_P2_transfer"
    ]
    comparison = {
        "Raw": {
            "graph_accuracy": raw[
                "graph"
            ]["accuracy"],
            "graph_f1": raw[
                "graph"
            ]["f1"],
            "graph_fpr": raw[
                "graph"
            ]["fpr"],
            "strict_all_task_exactness": raw[
                "strict"
            ][
                "strict_all_task_exactness"
            ],
        },
        "A0_frozen_P2_transfer": {
            "graph_accuracy": a0[
                "graph"
            ]["accuracy"],
            "graph_f1": a0[
                "graph"
            ]["f1"],
            "graph_fpr": a0[
                "graph"
            ]["fpr"],
            "strict_all_task_exactness": a0[
                "strict"
            ][
                "strict_all_task_exactness"
            ],
        },
        "A1_frozen_P2_exact_transfer": {
            "graph_accuracy": metrics[
                "graph"
            ]["accuracy"],
            "graph_f1": metrics[
                "graph"
            ]["f1"],
            "graph_fpr": metrics[
                "graph"
            ]["fpr"],
            "strict_all_task_exactness": metrics[
                "strict"
            ][
                "strict_all_task_exactness"
            ],
        },
        "A1_minus_Raw": {
            "graph_accuracy": (
                metrics["graph"][
                    "accuracy"
                ]
                - raw["graph"][
                    "accuracy"
                ]
            ),
            "graph_f1": (
                metrics["graph"]["f1"]
                - raw["graph"]["f1"]
            ),
            "graph_fpr": (
                metrics["graph"]["fpr"]
                - raw["graph"]["fpr"]
            ),
            "strict_all_task_exactness": (
                metrics["strict"][
                    "strict_all_task_exactness"
                ]
                - raw["strict"][
                    "strict_all_task_exactness"
                ]
            ),
        },
    }

    report = {
        "stage": STAGE,
        "status": "PASS",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "campaign_label": CAMPAIGN_LABEL,
        "classification": (
            "Tranche-A validation preliminary "
            "frozen-decoder transfer diagnostic"
        ),
        "audit_script_revision": (
            "v2_union_role_semantics_legality_correction"
        ),
        "scope": (
            "Full resumable certified exact-A1 "
            "decoding of all 13,863 immutable "
            "A-validation outputs using the "
            "unchanged P2 decoder and margin."
        ),
        "legality_semantics_correction": {
            "reason": (
                "source, transit and victim masks are unions across "
                "selected routes; endpoint/transit overlap can therefore "
                "be legal across different routes"
            ),
            "decoder_modified": False,
            "margin_modified": False,
            "previous_hash_valid_chunks_reusable": True,
        },
        "execution": {
            "items": EXPECTED_ITEMS,
            "chunk_size": CHUNK_SIZE,
            "chunk_count": total_chunks,
            "decoded_chunks_this_run": (
                completed_chunks
            ),
            "reused_chunks_this_run": (
                reused_chunks
            ),
            "all_chunks_hash_valid": True,
            "execution_contract_path": str(
                contract_path
            ),
            "execution_contract_sha256": (
                sha256_file(
                    contract_path
                )
            ),
            "schema_path": str(
                schema_path
            ),
            "schema_sha256": (
                sha256_file(schema_path)
            ),
        },
        "policy": {
            "decoder": (
                "frozen certified P2 exact "
                "XY legal decoder"
            ),
            "decoder_path": str(
                decoder_path
            ),
            "decoder_sha256": decoder_sha,
            "working_matches_canonical_preservation": (
                True
            ),
            "A1_margin_threshold": (
                A1_MARGIN_THRESHOLD
            ),
            "margin_retuned": False,
            "beam_decoder_selected": False,
        },
        "metrics": metrics,
        "comparison": comparison,
        "outputs": {
            "path": str(
                final_outputs_path
            ),
            "sha256": sha256_file(
                final_outputs_path
            ),
            "size_bytes": (
                final_outputs_path.stat().st_size
            ),
        },
        "decision": {
            "A1_exact_validation_complete": (
                True
            ),
            "A1_route_legality_certified": (
                True
            ),
            "A1_selected_as_P3_operating_policy": (
                False
            ),
            "reason_not_selected_yet": (
                "D4 is a frozen P2 policy "
                "transfer diagnostic. D5 must "
                "review Raw/A0/A1 performance "
                "and freeze the 4x4 expert "
                "interface before the 8x8 wrapper."
            ),
            "threshold_tuning_authorized": False,
            "additional_A_seed_authorized": False,
            "sealed_test_evaluation_authorized": (
                False
            ),
            "next_stage": (
                "V5_P3_D5_TRANCHE_A_DECODER_"
                "REVIEW_AND_4X4_EXPERT_INTERFACE_FREEZE"
            ),
        },
        "sealed_test": {
            "test_dataset_instantiated": False,
            "test_length_computed": False,
            "test_tensor_loaded": False,
            "evaluation_authorized": False,
        },
        "provenance": {
            "D3_report_sha256": (
                sha256_file(
                    d3_report_path
                )
            ),
            "D3_lock_sha256": (
                sha256_file(
                    d3_lock_path
                )
            ),
            "D2_report_sha256": (
                sha256_file(
                    d2_report_path
                )
            ),
            "D2_lock_sha256": (
                sha256_file(
                    d2_lock_path
                )
            ),
            "D1_export_sha256": export_sha,
            "installed_script_sha256": (
                sha256_file(
                    installed_script
                )
            ),
        },
        "model_replayed": False,
        "model_trained": False,
        "threshold_tuning_performed": False,
        "certified_dataset_modified": False,
        "generalization_claim_authorized": False,
    }
    atomic_json(
        report_path,
        report,
    )

    lock = {
        "stage": STAGE,
        "status": "PASS",
        "report_sha256": sha256_file(
            report_path
        ),
        "final_outputs_sha256": (
            sha256_file(
                final_outputs_path
            )
        ),
        "execution_contract_sha256": (
            sha256_file(
                contract_path
            )
        ),
        "schema_sha256": sha256_file(
            schema_path
        ),
        "D1_export_sha256": export_sha,
        "certified_decoder_sha256": (
            decoder_sha
        ),
        "A1_margin_threshold": (
            A1_MARGIN_THRESHOLD
        ),
        "items": EXPECTED_ITEMS,
        "chunk_size": CHUNK_SIZE,
        "chunk_count": total_chunks,
        "structural_legality_rate": (
            metrics["legality"][
                "structural_legality_rate"
            ]
        ),
        "test_tensor_loaded": False,
        "certified_dataset_modified": False,
    }
    atomic_json(
        lock_path,
        lock,
    )

    complete_path.write_text(
        f"{STAGE}_COMPLETE\n",
        encoding="utf-8",
    )
    if failure_path.exists():
        failure_path.unlink()

    print(f"{STAGE}_COMPLETE")
    print("status=PASS")
    print(
        "audit_script_revision="
        "v2_union_role_semantics_legality_correction"
    )
    print(
        f"campaign_label={CAMPAIGN_LABEL}"
    )
    print(f"items={EXPECTED_ITEMS}")
    print(f"chunk_size={CHUNK_SIZE}")
    print(
        f"chunk_count={total_chunks}"
    )
    print(
        "decoded_chunks_this_run="
        f"{completed_chunks}"
    )
    print(
        "reused_chunks_this_run="
        f"{reused_chunks}"
    )
    print(
        "all_chunks_hash_valid=true"
    )
    print(
        "A1_graph_accuracy="
        f"{metrics['graph']['accuracy']:.8f}"
    )
    print(
        "A1_graph_f1="
        f"{metrics['graph']['f1']:.8f}"
    )
    print(
        "A1_graph_fpr="
        f"{metrics['graph']['fpr']:.8f}"
    )
    print(
        "A1_margin_auroc="
        f"{metrics['graph']['margin_auroc']:.8f}"
    )
    print(
        "A1_margin_average_precision="
        f"{metrics['graph']['margin_average_precision']:.8f}"
    )
    print(
        "A1_count_active_macro_f1="
        f"{metrics['count_active']['macro_f1']:.8f}"
    )
    for role in (
        "source",
        "transit",
        "victim",
        "path",
    ):
        print(
            f"A1_{role}_exact_active="
            f"{metrics['roles'][role]['exact_active']:.8f}"
        )
    print(
        "A1_strict_all_task_exactness="
        f"{metrics['strict']['strict_all_task_exactness']:.8f}"
    )
    print(
        "strict_delta_A1_minus_Raw="
        f"{comparison['A1_minus_Raw']['strict_all_task_exactness']:.8f}"
    )
    print(
        "structural_legality_rate="
        f"{metrics['legality']['structural_legality_rate']:.8f}"
    )
    print(
        "source_victim_overlap_items="
        f"{metrics['legality']['source_victim_overlap_items']}"
    )
    print(
        "transit_endpoint_overlap_items="
        f"{metrics['legality']['transit_endpoint_overlap_items']}"
    )
    print(
        "mean_exact_decode_seconds="
        f"{metrics['timing']['mean_seconds']:.6f}"
    )
    print(
        "p95_exact_decode_seconds="
        f"{metrics['timing']['p95_seconds']:.6f}"
    )
    print(
        "p99_exact_decode_seconds="
        f"{metrics['timing']['p99_seconds']:.6f}"
    )
    print(
        "maximum_exact_decode_seconds="
        f"{metrics['timing']['maximum_seconds']:.6f}"
    )
    print(
        "threshold_tuning_performed=false"
    )
    print(
        "A1_margin_retuning_authorized=false"
    )
    print("beam_decoder_selected=false")
    print(
        "test_dataset_instantiated=false"
    )
    print(
        "test_length_computed=false"
    )
    print("test_tensor_loaded=false")
    print(
        "sealed_test_evaluation_authorized=false"
    )
    print(
        "next_stage="
        "V5_P3_D5_TRANCHE_A_DECODER_"
        "REVIEW_AND_4X4_EXPERT_INTERFACE_FREEZE"
    )
    print(
        f"final_outputs={final_outputs_path}"
    )
    print(f"report={report_path}")
    print(f"lock={lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
