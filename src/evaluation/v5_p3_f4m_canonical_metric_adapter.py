from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


SELECTION_COMPONENT_KEYS = (
    "graph_auroc",
    "graph_ap",
    "source_ap",
    "transit_ap",
    "victim_ap",
    "path_ap",
    "count_active_macro_f1",
)

ROLE_LABEL_KEYS = {
    "source": "y_source",
    "transit": "y_transit",
    "victim": "y_victim",
    "path": "y_attack_path",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def binary_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_pred = np.asarray(y_pred).astype(np.int64).reshape(-1)
    negative = y_true == 0
    denominator = int(np.sum(negative))
    require(denominator > 0, "FPR undefined: no negative examples")
    false_positive = int(np.sum((y_pred == 1) & negative))
    return false_positive / denominator


def exact_active(
    y_attack: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> float:
    y_attack = np.asarray(y_attack).astype(np.int64).reshape(-1)
    target = np.asarray(target).astype(bool)
    prediction = np.asarray(prediction).astype(bool)

    require(target.shape == prediction.shape, "role exact shape mismatch")
    active = y_attack == 1
    require(np.any(active), "exact-active undefined: no active samples")

    # Exact match across all 16 routers for every active validation item.
    per_item_exact = np.all(
        target[active] == prediction[active],
        axis=1,
    )
    return float(np.mean(per_item_exact))


def selection_score_from_metrics(metrics: dict[str, float]) -> float:
    missing = [
        key for key in SELECTION_COMPONENT_KEYS
        if key not in metrics
    ]
    require(not missing, f"selection components missing: {missing}")

    values = np.asarray(
        [float(metrics[key]) for key in SELECTION_COMPONENT_KEYS],
        dtype=np.float64,
    )
    require(np.all(np.isfinite(values)), "selection components are nonfinite")
    return float(np.mean(values, dtype=np.float64))


def compute_metrics_from_arrays(
    arrays: dict[str, np.ndarray],
    logit_keys: dict[str, str],
) -> dict[str, float]:
    required_heads = {
        "graph",
        "count",
        "source",
        "transit",
        "victim",
        "path",
    }
    require(
        set(logit_keys) == required_heads,
        f"unexpected logit-head mapping: {sorted(logit_keys)}",
    )

    y_attack = np.asarray(arrays["y_attack"]).astype(np.int64).reshape(-1)
    attack_logits = np.asarray(
        arrays[logit_keys["graph"]],
        dtype=np.float64,
    ).reshape(-1)
    attack_prediction = (attack_logits >= 0.0).astype(np.int64)

    metrics: dict[str, float] = {
        "graph_auroc": float(
            roc_auc_score(y_attack, attack_logits)
        ),
        "graph_ap": float(
            average_precision_score(y_attack, attack_logits)
        ),
        "graph_f1_at_0_5": float(
            f1_score(
                y_attack,
                attack_prediction,
                zero_division=0,
            )
        ),
        "graph_fpr_at_0_5": float(
            binary_fpr(y_attack, attack_prediction)
        ),
    }

    active = y_attack == 1
    require(np.any(active), "count metric undefined: no active samples")

    count_logits = np.asarray(arrays[logit_keys["count"]])
    require(
        count_logits.ndim == 2 and count_logits.shape[1] == 4,
        f"unexpected count-logit shape: {count_logits.shape}",
    )
    count_prediction = np.argmax(count_logits, axis=1).astype(np.int64)
    count_target = (
        np.asarray(arrays["y_attacker_count"])
        .astype(np.int64)
        .reshape(-1)
        - 1
    )

    metrics["count_active_macro_f1"] = float(
        f1_score(
            count_target[active],
            count_prediction[active],
            average="macro",
            labels=[0, 1, 2, 3],
            zero_division=0,
        )
    )

    for role, label_key in ROLE_LABEL_KEYS.items():
        target = np.asarray(arrays[label_key]).astype(np.int64)
        logits = np.asarray(
            arrays[logit_keys[role]],
            dtype=np.float64,
        )
        require(
            target.shape == logits.shape,
            f"{role} target/logit shape mismatch",
        )
        prediction = (logits >= 0.0).astype(np.int64)

        metrics[f"{role}_ap"] = float(
            average_precision_score(
                target.reshape(-1),
                logits.reshape(-1),
            )
        )
        metrics[f"{role}_exact_active"] = exact_active(
            y_attack,
            target,
            prediction,
        )

    metrics["selection_score"] = selection_score_from_metrics(metrics)
    return metrics


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: np.asarray(archive[key])
            for key in archive.files
        }


def compute_metrics_from_npz(
    npz_path: Path,
    logit_keys: dict[str, str],
) -> dict[str, float]:
    return compute_metrics_from_arrays(
        load_npz_arrays(npz_path),
        logit_keys,
    )


def adapter_contract() -> dict[str, Any]:
    return {
        "selection_formula": (
            "arithmetic mean of graph_auroc, graph_ap, source_ap, "
            "transit_ap, victim_ap, path_ap, count_active_macro_f1"
        ),
        "selection_component_keys": list(SELECTION_COMPONENT_KEYS),
        "graph_threshold": 0.0,
        "reporting_probability_threshold": 0.5,
        "count_label_transform": "y_attacker_count - 1",
        "count_macro_labels": [0, 1, 2, 3],
        "role_ap_scope": "all validation samples x all 16 routers",
        "role_exact_scope": "active samples x all 16 routers",
    }
