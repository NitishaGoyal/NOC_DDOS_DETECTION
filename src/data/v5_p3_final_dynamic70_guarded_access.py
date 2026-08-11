"""Guarded access boundary for the V5-P3-1500-D70 final A+B dataset.

This module intentionally exposes train and validation run tensors only.
The sealed test split is denied before directory enumeration or tensor
deserialization.

The final native dataset loader remains a separate, dataset-supplied artifact.
A later governed stage may integrate it behind this split guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class SealedTestAccessError(PermissionError):
    """Raised whenever the sealed final test split is requested."""


class InvalidSplitError(ValueError):
    """Raised for unknown split names."""


_ALLOWED_SPLITS = {
    "train": "train",
    "validation": "validation",
    "val": "validation",
}
_SEALED_ALIASES = {"test", "sealed_test", "final_test", "a_test", "b_test"}


def normalize_allowed_split(split: str) -> str:
    normalized = str(split).strip().lower().replace("-", "_")
    if normalized in _SEALED_ALIASES:
        raise SealedTestAccessError(
            "V5-P3 final test access is sealed. This guarded interface "
            "authorizes only train and validation."
        )
    if normalized not in _ALLOWED_SPLITS:
        raise InvalidSplitError(
            f"unsupported split {split!r}; allowed: train, validation"
        )
    return _ALLOWED_SPLITS[normalized]


@dataclass(frozen=True)
class GuardedFinalDynamic70Access:
    root: Path

    def __init__(self, root: str | Path) -> None:
        resolved = Path(root).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        object.__setattr__(self, "root", resolved)

    def split_dir(self, split: str) -> Path:
        canonical = normalize_allowed_split(split)
        path = self.root / "runs" / canonical
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path

    def list_runs(
        self,
        split: str,
        *,
        tranche: str | None = None,
        label: str | None = None,
    ) -> tuple[Path, ...]:
        directory = self.split_dir(split)

        tranche_prefix = None
        if tranche is not None:
            normalized_tranche = str(tranche).strip().upper()
            if normalized_tranche not in {"A", "B"}:
                raise ValueError("tranche must be A or B")
            tranche_prefix = f"P3{normalized_tranche}"

        label_suffix = None
        if label is not None:
            normalized_label = str(label).strip().upper()
            if normalized_label not in {"ATTACK", "CONTROL"}:
                raise ValueError("label must be ATTACK or CONTROL")
            label_suffix = f"_{normalized_label}.pt"

        selected: list[Path] = []
        for path in sorted(directory.glob("*.pt")):
            name = path.name
            if tranche_prefix is not None and not name.startswith(
                tranche_prefix
            ):
                continue
            if label_suffix is not None and not name.endswith(label_suffix):
                continue
            selected.append(path)
        return tuple(selected)

    def resolve_run(self, split: str, run_name: str) -> Path:
        directory = self.split_dir(split)
        name = Path(str(run_name)).name
        if name != str(run_name):
            raise ValueError("run_name must be a basename, not a path")
        if not name.endswith(".pt"):
            raise ValueError("run_name must end in .pt")
        path = (directory / name).resolve(strict=True)
        if path.parent != directory.resolve(strict=True):
            raise ValueError("run path escaped the authorized split")
        return path

    def load_run(
        self,
        split: str,
        run_name: str,
        *,
        map_location: str = "cpu",
    ) -> Any:
        path = self.resolve_run(split, run_name)
        import torch

        try:
            return torch.load(
                path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            return torch.load(path, map_location=map_location)

    def support_file(self, filename: str) -> Path:
        name = Path(str(filename)).name
        if name != str(filename):
            raise ValueError("support filename must be a basename")
        path = (self.root / name).resolve(strict=True)
        if path.parent != self.root:
            raise ValueError("support path escaped dataset root")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


def assert_test_is_sealed() -> None:
    try:
        normalize_allowed_split("test")
    except SealedTestAccessError:
        return
    raise AssertionError("sealed test split was unexpectedly authorized")
