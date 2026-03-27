#!/usr/bin/env python3
"""
Plot 2D (D=2) datasets from data/ as scatter plots.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_d2_datasets(data_dir: Path):
    pattern = re.compile(r"data_n(\d+)_k(\d+)_d2\.npy", re.IGNORECASE)
    for f in data_dir.glob("*.npy"):
        m = pattern.match(f.name)
        if m:
            yield f, int(m.group(1)), int(m.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2D datasets from data/ directory.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing .npy datasets")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save plot images")
    parser.add_argument("--max-points", type=int, default=50_000, help="Max points to plot")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    d2_files = list(find_d2_datasets(data_dir))
    if not d2_files:
        print(f"No D=2 datasets found in {data_dir}")
        return

    for path, n, k in d2_files:
        X = np.load(path)
        if X.ndim != 2 or X.shape[1] != 2:
            print(f"Skip {path.name}: shape {X.shape}")
            continue
        if len(X) > args.max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), args.max_points, replace=False)
            X_plot = X[idx]
            suffix = f" (subsampled to {args.max_points:,})"
        else:
            X_plot = X
            suffix = ""

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_plot[:, 0], X_plot[:, 1], s=0.5, alpha=0.6)
        ax.set_xlabel("Dimension 0")
        ax.set_ylabel("Dimension 1")
        ax.set_title(f"Dataset N={n:,}, K={k}, D=2{suffix}")
        ax.set_aspect("equal")
        fig.tight_layout()
        out_name = f"dataset_2d_n{n}_k{k}.png"
        fig.savefig(out_dir / out_name, dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / out_name}")

    print(f"Plotted {len(d2_files)} D=2 dataset(s) in {out_dir}")


if __name__ == "__main__":
    main()
