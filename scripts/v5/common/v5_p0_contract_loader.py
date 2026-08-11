#!/usr/bin/env python3
"""
Frozen V5 P0 contract-aware temporal-window loader.

The supplied tensor package remains immutable. This wrapper:
- delegates window construction to the supplied V5P0WindowDataset;
- uses stored x directly without another transform;
- selects one frozen feature variant;
- supplies the audited topology-derived raw Boolean physical-port mask;
- never adds router IDs, coordinates, pair metadata, or split metadata to x.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


ALL81 = list(range(81))
DYNAMIC70 = list(range(0, 30)) + list(range(31, 71))
PRIMARY58 = (
    list(range(0, 30))
    + list(range(31, 56))
    + [62, 64, 65]
)

FEATURE_VARIANTS = {
    "ALL81_COMPATIBILITY": ALL81,
    "DYNAMIC70": DYNAMIC70,
    "PRIMARY58": PRIMARY58,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_supplied_dataset_class(root: Path):
    loader_path = root / "dataset_loader.py"
    if not loader_path.is_file():
        raise FileNotFoundError(f"supplied loader missing: {loader_path}")

    spec = importlib.util.spec_from_file_location(
        "v5_p0_supplied_dataset_loader",
        loader_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import supplied loader: {loader_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset_class = getattr(module, "V5P0WindowDataset", None)
    if dataset_class is None:
        raise AttributeError(
            "supplied loader does not define V5P0WindowDataset"
        )
    return dataset_class


class V5P0ContractWindowDataset(Dataset):
    """
    Contract-aware wrapper around the supplied V5P0WindowDataset.

    Returned x layout:
        [16, F, window]

    Default primary layout:
        [16, 58, 32]

    Added structural tensor:
        physical_port_mask [16, 10] torch.bool
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        contract_dir: str | Path,
        mask_audit_dir: str | Path,
        window: int = 32,
        stride: int = 8,
        active_only: bool = False,
        feature_variant: str = "PRIMARY58",
    ) -> None:
        super().__init__()

        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.contract_dir = Path(contract_dir).expanduser().resolve()
        self.mask_audit_dir = Path(mask_audit_dir).expanduser().resolve()
        self.window = int(window)
        self.stride = int(stride)
        self.active_only = bool(active_only)
        self.feature_variant = str(feature_variant)

        if self.feature_variant not in FEATURE_VARIANTS:
            raise ValueError(
                f"unknown feature_variant={self.feature_variant!r}; "
                f"valid={sorted(FEATURE_VARIANTS)}"
            )
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        contract_path = (
            self.contract_dir
            / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT.json"
        )
        pass_marker = (
            self.contract_dir
            / "V5_P0_A2_FEATURE_AND_NORMALIZATION_CONTRACT_PASS"
        )
        if not pass_marker.is_file():
            raise RuntimeError(
                f"A2 PASS marker missing: {pass_marker}"
            )
        if not contract_path.is_file():
            raise FileNotFoundError(
                f"A2 contract missing: {contract_path}"
            )

        contract = _load_json(contract_path)
        if contract.get("status") != "PASS":
            raise RuntimeError("A2 feature contract is not PASS")
        if contract.get("contract_version") != "V5_P0_A2_D0":
            raise RuntimeError(
                "unexpected A2 contract version: "
                f"{contract.get('contract_version')!r}"
            )

        frozen_variant = contract["feature_variants"][
            self.feature_variant
        ]
        frozen_indices = [
            int(index)
            for index in frozen_variant["indices"]
        ]
        expected_indices = FEATURE_VARIANTS[self.feature_variant]
        if frozen_indices != expected_indices:
            raise RuntimeError(
                f"{self.feature_variant} indices disagree with "
                "the frozen implementation"
            )

        normalization = contract["normalization"]
        if not bool(
            normalization.get("stored_x_is_already_transformed")
        ):
            raise RuntimeError(
                "contract does not mark stored x as transformed"
            )
        if bool(
            normalization.get("apply_log1p_in_loader")
        ):
            raise RuntimeError(
                "contract unexpectedly requests log1p in loader"
            )
        if bool(
            normalization.get("apply_normalization_pt_in_loader")
        ):
            raise RuntimeError(
                "contract unexpectedly requests normalization in loader"
            )

        raw_mask_path = (
            self.mask_audit_dir
            / "topology_derived_raw_physical_port_mask.pt"
        )
        if not raw_mask_path.is_file():
            raise FileNotFoundError(
                f"audited raw mask missing: {raw_mask_path}"
            )

        mask_payload = torch.load(
            raw_mask_path,
            map_location="cpu",
            weights_only=False,
        )
        physical_mask = mask_payload.get("physical_port_mask")
        if not isinstance(physical_mask, torch.Tensor):
            raise TypeError(
                "mask artifact does not contain physical_port_mask tensor"
            )
        physical_mask = physical_mask.detach().cpu()
        if tuple(physical_mask.shape) != (16, 10):
            raise ValueError(
                "physical_port_mask shape="
                f"{tuple(physical_mask.shape)}, expected (16,10)"
            )
        if physical_mask.dtype != torch.bool:
            raise TypeError(
                "physical_port_mask dtype="
                f"{physical_mask.dtype}, expected torch.bool"
            )

        supplied_dataset_class = _load_supplied_dataset_class(
            self.root
        )
        self.base_dataset = supplied_dataset_class(
            root=self.root,
            split=self.split,
            window=self.window,
            stride=self.stride,
            active_only=self.active_only,
        )

        self.feature_indices = tuple(expected_indices)
        self.feature_count = len(self.feature_indices)
        self.physical_port_mask = physical_mask.contiguous()
        self.contract_version = "V5_P0_A2_D0"
        self.normalization_applied_by_wrapper = False

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_item = self.base_dataset[index]
        if not isinstance(base_item, dict):
            raise TypeError(
                "supplied dataset item is not a dictionary"
            )
        if "x" not in base_item:
            raise KeyError("supplied dataset item lacks x")

        x = base_item["x"]
        if not isinstance(x, torch.Tensor):
            raise TypeError("supplied x is not a tensor")
        if x.ndim != 3:
            raise ValueError(
                f"supplied x rank={x.ndim}, expected 3"
            )
        if tuple(x.shape) != (16, 81, self.window):
            raise ValueError(
                f"supplied x shape={tuple(x.shape)}, expected "
                f"(16,81,{self.window})"
            )
        if x.is_floating_point() and not bool(
            torch.isfinite(x).all()
        ):
            raise ValueError("supplied x contains non-finite values")

        selected_x = x[:, self.feature_indices, :].contiguous()
        expected_shape = (
            16,
            self.feature_count,
            self.window,
        )
        if tuple(selected_x.shape) != expected_shape:
            raise RuntimeError(
                f"selected x shape={tuple(selected_x.shape)}, "
                f"expected={expected_shape}"
            )

        item = dict(base_item)
        item["x"] = selected_x
        item["physical_port_mask"] = self.physical_port_mask
        return item
