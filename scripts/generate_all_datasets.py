#!/usr/bin/env python3
"""
Generate all datasets required for benchmark sweeps and save to ../data.
"""

from pathlib import Path

import numpy as np

from generate_dataset import generate_dataset

DATASET_CONFIGS = [
    (10_000, 50, 32),
    (100_000, 50, 32),
    (1_000_000, 50, 32),
    (5_000_000, 50, 32),
    (10_000_000, 50, 32),
    (1_000_000, 10, 32),
    (1_000_000, 100, 32),
    (1_000_000, 200, 32),
    (1_000_000, 500, 32),
    (1_000_000, 50, 2),
    (1_000_000, 50, 8),
    (1_000_000, 50, 64),
    (1_000_000, 50, 128),
    (1_000_000, 50, 256),
    (1_000_000, 50, 512),
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for n, k, d in DATASET_CONFIGS:
        path = data_dir / f"data_n{n}_k{k}_d{d}.npy"
        if path.exists():
            print(f"Skip (exists): {path.name}")
            continue
        print(f"Generating N={n:,} K={k} D={d} ...")
        X = generate_dataset(n, k, d)
        np.save(path, X)
        print(f"  Saved {X.shape} -> {path}")

    print(f"Done. Datasets in {data_dir}")


if __name__ == "__main__":
    main()
