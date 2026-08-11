#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-data", required=True)
    parser.add_argument("--out-data", required=True)
    parser.add_argument("--count-clip", type=float, default=256.0)
    args = parser.parse_args()

    d = np.load(args.in_data, allow_pickle=True)

    x_ifd = d["x"].astype(np.float32)  # [S, 16, 8, 2]
    counts = d["flit_count_window"].astype(np.float32)  # [S, 16, 8, 2]

    counts_norm = np.clip(counts / args.count_clip, 0.0, 1.0).astype(np.float32)

    x4 = np.concatenate([x_ifd, counts_norm], axis=-1).astype(np.float32)

    out = Path(args.out_data)
    out.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {k: d[k] for k in d.files}
    save_dict["x"] = x4
    save_dict["feature_names"] = np.array([
        "ifd_in_norm",
        "ifd_out_norm",
        "input_flit_count_norm",
        "output_flit_count_norm",
    ])
    save_dict["count_clip"] = np.array([args.count_clip], dtype=np.float32)

    np.savez_compressed(out, **save_dict)

    print("input:", args.in_data)
    print("output:", out)
    print("old x shape:", x_ifd.shape)
    print("new x shape:", x4.shape)
    print("count clip:", args.count_clip)
    print("x min/max:", float(x4.min()), float(x4.max()))
    print("count norm min/max:", float(counts_norm.min()), float(counts_norm.max()))
    print("feature names:", save_dict["feature_names"].tolist())


if __name__ == "__main__":
    main()
