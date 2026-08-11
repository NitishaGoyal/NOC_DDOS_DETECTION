#!/usr/bin/env python3
"""
Create the Chrono-B1 training script from the frozen Chrono-A1 source.

The patch is deliberately narrow:
- imports ScenarioBalancedBatchSampler;
- adds sampler-only CLI arguments;
- replaces the shuffled train loader with the B1 batch sampler;
- adds a sequential train-evaluation loader;
- reseeds the sampler once per epoch;
- logs sampler statistics;
- evaluates training metrics on the original full training split;
- records sampler configuration in summary.json.

Model classes, forward pass, losses, class weights, optimizer, validation
loader, test loader, checkpoint score, and early stopping are not changed.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one source match, found {count}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()

    if not source.is_file():
        raise SystemExit(f"STOP: source does not exist: {source}")
    if destination.exists():
        raise SystemExit(f"STOP: destination already exists: {destination}")

    actual_hash = sha256(source)
    if actual_hash != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "STOP: Chrono-A1 source hash mismatch\n"
            f"expected={EXPECTED_SOURCE_SHA256}\n"
            f"actual={actual_hash}"
        )

    text = source.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "from torch.utils.data import Dataset, DataLoader\n",
        (
            "from torch.utils.data import Dataset, DataLoader\n"
            "from samplers.chrono_b1_scenario_sampler import "
            "ScenarioBalancedBatchSampler\n"
        ),
        "sampler import",
    )

    text = replace_once(
        text,
        '    parser.add_argument("--seed", type=int, default=7)\n',
        (
            '    parser.add_argument("--seed", type=int, default=7)\n'
            '    parser.add_argument("--b1-samples-per-epoch", '
            'type=int, default=148185)\n'
            '    parser.add_argument("--b1-max-temporal-retries", '
            'type=int, default=32)\n'
        ),
        "sampler CLI arguments",
    )

    text = replace_once(
        text,
        (
            "    if args.min_delta < 0:\n"
            '        raise ValueError("--min-delta cannot be negative.")\n'
        ),
        (
            "    if args.min_delta < 0:\n"
            '        raise ValueError("--min-delta cannot be negative.")\n'
            "    if args.b1_samples_per_epoch <= 0:\n"
            '        raise ValueError("--b1-samples-per-epoch must be positive.")\n'
            "    if args.b1_max_temporal_retries <= 0:\n"
            '        raise ValueError("--b1-max-temporal-retries must be positive.")\n'
        ),
        "sampler argument validation",
    )

    text = replace_once(
        text,
        (
            "    train_loader = DataLoader(train_dataset, "
            "batch_size=args.batch_size, shuffle=True, num_workers=0)\n"
            "    val_loader = DataLoader(val_dataset, "
            "batch_size=args.batch_size, shuffle=False, num_workers=0)\n"
            "    test_loader = DataLoader(test_dataset, "
            "batch_size=args.batch_size, shuffle=False, num_workers=0)\n"
        ),
        (
            "    train_batch_sampler = ScenarioBalancedBatchSampler(\n"
            "        y_graph=data[\"y_graph\"][train_idx],\n"
            "        profile=data[\"profile\"][train_idx],\n"
            "        active_cores=data[\"active_cores\"][train_idx],\n"
            "        strength=data[\"strength\"][train_idx],\n"
            "        attackers=data[\"attackers\"][train_idx],\n"
            "        run_id=data[\"run_id\"][train_idx],\n"
            "        end_epoch=data[\"end_epoch\"][train_idx],\n"
            "        batch_size=args.batch_size,\n"
            "        samples_per_epoch=args.b1_samples_per_epoch,\n"
            "        temporal_window_length=int(x_shape[2]),\n"
            "        seed=args.seed,\n"
            "        drop_last=False,\n"
            "        max_temporal_retries=args.b1_max_temporal_retries,\n"
            "    )\n"
            "    train_loader = DataLoader(\n"
            "        train_dataset,\n"
            "        batch_sampler=train_batch_sampler,\n"
            "        num_workers=0,\n"
            "    )\n"
            "    train_eval_loader = DataLoader(\n"
            "        train_dataset,\n"
            "        batch_size=args.batch_size,\n"
            "        shuffle=False,\n"
            "        num_workers=0,\n"
            "    )\n"
            "    val_loader = DataLoader(val_dataset, "
            "batch_size=args.batch_size, shuffle=False, num_workers=0)\n"
            "    test_loader = DataLoader(test_dataset, "
            "batch_size=args.batch_size, shuffle=False, num_workers=0)\n"
            "\n"
            "    print(\"\\nChrono-B1 sampler:\")\n"
            "    print(\"  samples per epoch:\", args.b1_samples_per_epoch)\n"
            "    print(\"  graph class target: 50% normal / 50% attack\")\n"
            "    print(\"  temporal window gap:\", int(x_shape[2]))\n"
        ),
        "training loader replacement",
    )

    text = replace_once(
        text,
        (
            "    for epoch in range(1, args.epochs + 1):\n"
            "        model.train()\n"
        ),
        (
            "    for epoch in range(1, args.epochs + 1):\n"
            "        train_batch_sampler.set_epoch(epoch - 1)\n"
            "        model.train()\n"
        ),
        "epoch sampler reseed",
    )

    text = replace_once(
        text,
        (
            "        train_step_loss = total_loss / max(1, batches)\n"
            "        eval_kwargs = {\n"
        ),
        (
            "        train_step_loss = total_loss / max(1, batches)\n"
            "        sampler_stats = train_batch_sampler.last_epoch_stats()\n"
            "        eval_kwargs = {\n"
        ),
        "sampler epoch statistics",
    )

    text = replace_once(
        text,
        "        train_metrics = evaluate(model, train_loader, **eval_kwargs)\n",
        (
            "        train_metrics = evaluate("
            "model, train_eval_loader, **eval_kwargs)\n"
        ),
        "per-epoch full train evaluation",
    )

    text = replace_once(
        text,
        (
            '            "epochs_without_improvement": '
            "int(epochs_without_improvement),\n"
            "        })\n"
        ),
        (
            '            "epochs_without_improvement": '
            "int(epochs_without_improvement),\n"
            '            "sampler": sampler_stats,\n'
            "        })\n"
        ),
        "history sampler logging",
    )

    text = replace_once(
        text,
        "    train_metrics = evaluate(model, train_loader, **eval_kwargs)\n",
        (
            "    train_metrics = evaluate("
            "model, train_eval_loader, **eval_kwargs)\n"
        ),
        "final full train evaluation",
    )

    text = replace_once(
        text,
        (
            '        "training": {\n'
            '            "maximum_epochs": int(args.epochs),\n'
        ),
        (
            '        "sampling": {\n'
            '            "name": "Chrono-B1 hierarchical scenario-balanced",\n'
            '            "samples_per_epoch": '
            "int(args.b1_samples_per_epoch),\n"
            '            "replacement": True,\n'
            '            "normal_probability": 0.5,\n'
            '            "attack_probability": 0.5,\n'
            '            "temporal_window_length": int(x_shape[2]),\n'
            '            "max_temporal_retries": '
            "int(args.b1_max_temporal_retries),\n"
            '            "final_epoch_stats": '
            "train_batch_sampler.last_epoch_stats(),\n"
            "        },\n"
            '        "training": {\n'
            '            "maximum_epochs": int(args.epochs),\n'
        ),
        "summary sampler metadata",
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    destination.chmod(0o555)

    print(f"source_sha256={actual_hash}")
    print(f"destination_sha256={sha256(destination)}")
    print(f"CREATED: {destination}")


if __name__ == "__main__":
    main()
