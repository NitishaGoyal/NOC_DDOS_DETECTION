#!/usr/bin/env python3
"""
Leakage-safe P2 dataset loader for the frozen B3 Conv1D-only architecture.

Authorized splits in this module:
    train
    validation

The P2 test split is intentionally rejected. A separately locked test loader
must be created only after the P2 model, checkpoint, and thresholds are frozen.

Returned model/target dictionary:
    x                  float32 [16,58,32]
    physical_port_mask bool    [16,10]
    y_attack           float32 scalar
    y_attacker_count   int64   scalar
    y_source           float32 [16]
    y_transit          float32 [16]
    y_victim           float32 [16]
    y_attack_path      float32 [16]
    role_mask          uint8/bool [16]

No identifiers, metadata, edge_index, coordinates, run lengths, or window
positions are returned.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


PRIMARY58_INDICES = tuple(
    list(range(0, 30))
    + list(range(31, 56))
    + [62, 64, 65]
)
WINDOW = 32
STRIDE = 8
AUTHORIZED_SPLITS = {"train", "validation"}

MODEL_ITEM_KEYS = {
    "x",
    "physical_port_mask",
    "y_attack",
    "y_attacker_count",
    "y_source",
    "y_transit",
    "y_victim",
    "y_attack_path",
    "role_mask",
}


@dataclass(frozen=True)
class IndexEntry:
    file_path: Path
    pair_key: str
    mode: str
    start: int
    target: int
    common_length: int


def _as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid integer field {key!r} for "
            f"{row.get('split')}/{row.get('pair_key')}"
        ) from exc


def derive_corrected_physical_port_mask(
    topology: dict[str, Any],
) -> torch.Tensor:
    """
    Derive Boolean [16,10] in this exact order:
      input : local,north,east,south,west
      output: local,north,east,south,west

    Frozen A2-R2 coordinate semantics:
      vertical   = coord[0]
      horizontal = coord[1]
      north      = increasing coord[0]
      east       = increasing coord[1]
    """
    coordinates = topology.get("router_coordinates")
    if (
        not isinstance(coordinates, torch.Tensor)
        or coordinates.dtype != torch.int64
        or tuple(coordinates.shape) != (16, 2)
    ):
        raise ValueError(
            "topology router_coordinates must be int64 [16,2]"
        )

    vertical = coordinates[:, 0]
    horizontal = coordinates[:, 1]

    vertical_min = int(vertical.min().item())
    vertical_max = int(vertical.max().item())
    horizontal_min = int(horizontal.min().item())
    horizontal_max = int(horizontal.max().item())

    local = torch.ones_like(vertical, dtype=torch.bool)
    north = vertical < vertical_max
    east = horizontal < horizontal_max
    south = vertical > vertical_min
    west = horizontal > horizontal_min

    one_side = torch.stack(
        (local, north, east, south, west),
        dim=1,
    )
    mask = torch.cat((one_side, one_side), dim=1)

    if tuple(mask.shape) != (16, 10):
        raise RuntimeError("derived physical-port mask is not [16,10]")
    if mask.dtype != torch.bool:
        raise RuntimeError("derived physical-port mask is not Boolean")
    if int(mask.sum().item()) != 128:
        raise RuntimeError(
            f"derived physical-port mask true count is "
            f"{int(mask.sum().item())}, expected 128"
        )
    return mask.contiguous()


class V5P2PairAlignedPrimary58Dataset(Dataset):
    """
    Pair-aligned P2 train/validation windows for frozen B3.

    Window-index construction comes exclusively from the frozen A1-R2 pair
    manifest. Both ATTACK and CONTROL members of a matched pair receive the
    same ordered starts inside:

        common_length = min(T_attack, T_control)

    The stored x values are already log1p-transformed and train-standardized.
    This loader performs no normalization or inverse transformation.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        pair_manifest: str | Path,
        *,
        window: int = WINDOW,
        stride: int = STRIDE,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.pair_manifest = Path(pair_manifest).expanduser().resolve()
        self.window = int(window)
        self.stride = int(stride)

        if self.split not in AUTHORIZED_SPLITS:
            raise ValueError(
                "This audited loader authorizes only train and validation. "
                "The P2 test split remains locked."
            )
        if self.window != WINDOW:
            raise ValueError(
                f"window must remain frozen at {WINDOW}, got {self.window}"
            )
        if self.stride != STRIDE:
            raise ValueError(
                f"stride must remain frozen at {STRIDE}, got {self.stride}"
            )
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root missing: {self.root}")
        if not self.pair_manifest.is_file():
            raise FileNotFoundError(
                f"pair manifest missing: {self.pair_manifest}"
            )

        split_dir = self.root / "runs" / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"authorized split directory missing: {split_dir}"
            )

        topology = torch.load(
            self.root / "topology.pt",
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(topology, dict):
            raise TypeError("topology.pt payload must be a dictionary")
        self.physical_port_mask = (
            derive_corrected_physical_port_mask(topology)
        )

        self._index: list[IndexEntry] = []
        self._pair_count = 0

        with self.pair_manifest.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))

        seen_pairs: set[str] = set()
        for row in rows:
            if row.get("split") != self.split:
                continue

            pair_key = row.get("pair_key", "")
            if not pair_key:
                raise ValueError("pair manifest row has empty pair_key")
            if pair_key in seen_pairs:
                raise ValueError(
                    f"duplicate pair manifest row: "
                    f"{self.split}/{pair_key}"
                )
            seen_pairs.add(pair_key)
            self._pair_count += 1

            common_length = _as_int(row, "common_length")
            window_count = _as_int(
                row,
                "window_count_per_member",
            )
            manifest_window = _as_int(row, "window")
            manifest_stride = _as_int(row, "stride")

            if manifest_window != self.window:
                raise ValueError(
                    f"{self.split}/{pair_key}: manifest window="
                    f"{manifest_window}, expected {self.window}"
                )
            if manifest_stride != self.stride:
                raise ValueError(
                    f"{self.split}/{pair_key}: manifest stride="
                    f"{manifest_stride}, expected {self.stride}"
                )

            expected_window_count = (
                0
                if common_length < self.window
                else 1
                + (common_length - self.window) // self.stride
            )
            if window_count != expected_window_count:
                raise ValueError(
                    f"{self.split}/{pair_key}: manifest window count="
                    f"{window_count}, expected {expected_window_count}"
                )

            attack_path = split_dir / f"{pair_key}_ATTACK.pt"
            control_path = split_dir / f"{pair_key}_CONTROL.pt"
            if not attack_path.is_file():
                raise FileNotFoundError(
                    f"missing ATTACK tensor: {attack_path}"
                )
            if not control_path.is_file():
                raise FileNotFoundError(
                    f"missing CONTROL tensor: {control_path}"
                )

            for window_index in range(window_count):
                start = window_index * self.stride
                target = start + self.window - 1

                if target >= common_length:
                    raise ValueError(
                        f"{self.split}/{pair_key}: target {target} "
                        f"outside common length {common_length}"
                    )

                # Pair-major, start-major, ATTACK then CONTROL.
                self._index.append(
                    IndexEntry(
                        file_path=attack_path,
                        pair_key=pair_key,
                        mode="attack",
                        start=start,
                        target=target,
                        common_length=common_length,
                    )
                )
                self._index.append(
                    IndexEntry(
                        file_path=control_path,
                        pair_key=pair_key,
                        mode="control",
                        start=start,
                        target=target,
                        common_length=common_length,
                    )
                )

        if self._pair_count == 0:
            raise ValueError(
                f"pair manifest contains no rows for split {self.split!r}"
            )
        if not self._index:
            raise ValueError(
                f"no aligned windows produced for split {self.split!r}"
            )
        if len(self._index) % 2 != 0:
            raise RuntimeError(
                "pair-aligned dataset length must be even"
            )

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_run(path: Path) -> dict[str, Any]:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise TypeError(f"run payload is not a dictionary: {path}")
        return payload

    def __len__(self) -> int:
        return len(self._index)

    @property
    def pair_count(self) -> int:
        return self._pair_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self._index[index]
        run = self._load_run(entry.file_path)

        x_stored = run.get("x")
        if (
            not isinstance(x_stored, torch.Tensor)
            or x_stored.dtype != torch.float32
            or x_stored.ndim != 3
            or tuple(x_stored.shape[1:]) != (16, 81)
        ):
            raise ValueError(
                f"invalid stored x tensor in {entry.file_path}"
            )
        if x_stored.shape[0] < entry.common_length:
            raise ValueError(
                f"{entry.file_path} length={x_stored.shape[0]} "
                f"is below common length={entry.common_length}"
            )

        stop = entry.start + self.window
        x = (
            x_stored[
                entry.start:stop,
                :,
                PRIMARY58_INDICES,
            ]
            .permute(1, 2, 0)
            .contiguous()
        )

        if tuple(x.shape) != (16, 58, 32):
            raise RuntimeError(
                f"constructed x shape={tuple(x.shape)}, "
                "expected [16,58,32]"
            )

        item = {
            "x": x,
            "physical_port_mask": (
                self.physical_port_mask.clone()
            ),
            "y_attack": run["y_attack"][entry.target].clone(),
            "y_attacker_count": (
                run["y_attacker_count"][entry.target].clone()
            ),
            "y_source": run["y_source"][entry.target].clone(),
            "y_transit": run["y_transit"][entry.target].clone(),
            "y_victim": run["y_victim"][entry.target].clone(),
            "y_attack_path": (
                run["y_attack_path"][entry.target].clone()
            ),
            "role_mask": run["role_mask"][entry.target].clone(),
        }

        if set(item) != MODEL_ITEM_KEYS:
            raise RuntimeError("loader item-key contract changed")
        return item
