#!/usr/bin/env python3

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(data_path, allow_pickle=True)
    run_ids = d["run_id"].astype(str)
    y_graph = d["y_graph"].astype(np.int64)
    y_node = d["y_node"].astype(np.float32)

    unique_runs = sorted(set(run_ids.tolist()))

    normal_runs = []
    attack_runs_by_k = defaultdict(list)

    for run in unique_runs:
        idx = np.where(run_ids == run)[0]
        first = idx[0]
        graph_label = int(y_graph[first])
        k = int(y_node[first].sum())

        if graph_label == 0 or "-A-" not in run:
            normal_runs.append(run)
        else:
            attack_runs_by_k[k].append(run)

    rng = np.random.default_rng(args.seed)

    train_runs = []
    val_runs = []
    test_runs = []

    # Split normal run temporally into train/val/test
    train_idx, val_idx, test_idx = [], [], []

    for run in normal_runs:
        idx = np.sort(np.where(run_ids == run)[0])
        n = len(idx)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

        train_runs.append(run + " [normal temporal 70%]")
        val_runs.append(run + " [normal temporal 15%]")
        test_runs.append(run + " [normal temporal 15%]")

    # For each attacker count, assign one run to val, one to test, rest to train
    for k in sorted(attack_runs_by_k):
        runs = sorted(attack_runs_by_k[k])
        rng.shuffle(runs)

        if len(runs) < 3:
            raise RuntimeError(f"Need at least 3 runs for {k}-attacker group, found {len(runs)}")

        val_run = runs[0]
        test_run = runs[1]
        train_group = runs[2:]

        val_runs.append(val_run)
        test_runs.append(test_run)
        train_runs.extend(train_group)

        for run in train_group:
            train_idx.extend(np.sort(np.where(run_ids == run)[0]))
        val_idx.extend(np.sort(np.where(run_ids == val_run)[0]))
        test_idx.extend(np.sort(np.where(run_ids == test_run)[0]))

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    np.savez_compressed(
        out_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_runs=np.asarray(train_runs, dtype=str),
        val_runs=np.asarray(val_runs, dtype=str),
        test_runs=np.asarray(test_runs, dtype=str),
        seed=np.asarray([args.seed], dtype=np.int64),
    )

    lines = []
    lines.append("Balanced placement split")
    lines.append("=" * 80)
    lines.append(f"data: {data_path}")
    lines.append(f"out: {out_path}")
    lines.append(f"seed: {args.seed}")
    lines.append("")
    lines.append(f"train samples: {len(train_idx)}")
    lines.append(f"val samples:   {len(val_idx)}")
    lines.append(f"test samples:  {len(test_idx)}")
    lines.append("")

    for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        lines.append("")
        lines.append(split_name.upper())
        lines.append("-" * 80)

        split_runs = sorted(set(run_ids[split_idx].tolist()))
        for run in split_runs:
            idx = split_idx[run_ids[split_idx] == run]
            first_global_idx = np.where(run_ids == run)[0][0]
            k = int(y_node[first_global_idx].sum())
            g = int(y_graph[first_global_idx])
            lines.append(f"k={k} graph={g} n={len(idx):5d} run={run}")

    report_text = "\n".join(lines)
    print(report_text)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text)
        print()
        print("wrote report:", report_path)


if __name__ == "__main__":
    main()
