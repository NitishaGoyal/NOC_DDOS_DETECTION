"""Train/validation-only repository wrapper for V5-P3 Tranche-A."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class SealedTestAccessError(PermissionError):
    """Raised before any attempt to construct the sealed test dataset."""


class DatasetInterfaceError(RuntimeError):
    """Raised when the certified loader interface cannot be resolved."""


def _load_original_class(dataset_root: Path):
    loader_path = dataset_root / "dataset_loader.py"
    if not loader_path.is_file():
        raise DatasetInterfaceError(
            f"certified dataset_loader.py missing: {loader_path}"
        )

    module_name = (
        "_v5_p3_tranche_a_certified_loader_"
        + str(abs(hash(str(loader_path.resolve()))))
    )
    spec = importlib.util.spec_from_file_location(module_name, loader_path)
    if spec is None or spec.loader is None:
        raise DatasetInterfaceError(
            f"cannot import certified loader: {loader_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    class_name = "V5P3TrancheAPreliminaryDataset"
    dataset_class = getattr(module, class_name, None)
    if dataset_class is None:
        raise DatasetInterfaceError(
            f"{class_name} not found in {loader_path}"
        )
    return dataset_class


def _construct_original(
    dataset_class,
    dataset_root: Path,
    split: str,
    user_kwargs: dict[str, Any],
):
    signature = inspect.signature(dataset_class.__init__)
    parameters = signature.parameters
    kwargs = dict(user_kwargs)

    root_names = (
        "root",
        "data_root",
        "dataset_root",
        "root_dir",
        "base_dir",
        "dataset_dir",
    )
    split_names = ("split", "split_name", "partition")

    root_injected = False
    for name in root_names:
        if name in parameters:
            kwargs[name] = dataset_root
            root_injected = True
            break

    split_injected = False
    for name in split_names:
        if name in parameters:
            kwargs[name] = split
            split_injected = True
            break

    normalization_path = (
        dataset_root / "normalization_tranche_a_provisional.pt"
    )
    for name in (
        "normalization_path",
        "normalization_file",
        "norm_path",
    ):
        parameter = parameters.get(name)
        if (
            parameter is not None
            and name not in kwargs
            and parameter.default is inspect.Parameter.empty
        ):
            kwargs[name] = normalization_path
            break

    if root_injected and split_injected:
        return dataset_class(**kwargs)

    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    if not root_injected and split_injected:
        attempts.append(((dataset_root,), kwargs))
    elif root_injected and not split_injected:
        attempts.append(((split,), kwargs))
    else:
        attempts.extend(
            [
                ((dataset_root, split), kwargs),
                ((dataset_root,), {**kwargs, "split": split}),
            ]
        )

    errors = []
    for args, attempt_kwargs in attempts:
        try:
            return dataset_class(*args, **attempt_kwargs)
        except TypeError as exc:
            errors.append(str(exc))

    raise DatasetInterfaceError(
        "unable to construct certified dataset class; "
        f"signature={signature}; errors={errors}"
    )


class GuardedV5P3TrancheAPreliminaryDataset(Dataset):
    """Only exact train and validation split names are permitted."""

    ALLOWED_SPLITS = ("train", "validation")

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        **kwargs: Any,
    ) -> None:
        if not isinstance(split, str):
            raise TypeError("split must be a string")

        # Exact names only. This check happens before importing or
        # constructing the certified dataset class.
        if split not in self.ALLOWED_SPLITS:
            raise SealedTestAccessError(
                "V5-P3 Tranche-A test is sealed. Only split='train' "
                "and split='validation' are authorized."
            )

        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        dataset_class = _load_original_class(self.dataset_root)
        self._dataset = _construct_original(
            dataset_class,
            self.dataset_root,
            split,
            kwargs,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        return self._dataset[index]

    @property
    def original_dataset(self):
        return self._dataset
