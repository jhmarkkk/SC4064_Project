#!/usr/bin/env python3
"""
Generate synthetic float32 datasets for K-means benchmarking.
Uses make_blobs for moderate N; chunked generation for N >= 10M to avoid OOM.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs

CHUNK_SIZE = 2_000_000


def generate_blobs(n_samples: int, n_features: int, n_centers: int, seed: int = 42) -> np.ndarray:
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        random_state=seed,
    )
    return X.astype(np.float32)


def generate_chunked(n_samples: int, n_features: int, n_centers: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    remaining = n_samples
    while remaining > 0:
        size = min(CHUNK_SIZE, remaining)
        chunk = rng.standard_normal((size, n_features)).astype(np.float32)
        chunks.append(chunk)
        remaining -= size
    return np.vstack(chunks)


def generate_dataset(
    n: int,
    k: int,
    d: int,
    seed: int = 42,
    use_chunked_threshold: int = 10_000_000,
) -> np.ndarray:
    if n >= use_chunked_threshold:
        return generate_chunked(n, d, min(k, n), seed)
    return generate_blobs(n, d, min(k, n), seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic float32 datasets for K-means benchmarking.")
    parser.add_argument("--n", type=int, required=True, help="Number of data points")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters")
    parser.add_argument("--d", type=int, required=True, help="Number of dimensions")
    parser.add_argument("--out", type=str, required=True, help="Output path: directory or .npy file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.n <= 0 or args.k <= 0 or args.d <= 0:
        raise ValueError("n, k, d must be positive")

    X = generate_dataset(args.n, args.k, args.d, seed=args.seed)
    out = Path(args.out)
    if out.suffix.lower() == ".npy":
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, X)
        print(f"Saved {X.shape} float32 array to {out}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"data_n{args.n}_k{args.k}_d{args.d}.npy"
        np.save(path, X)
        print(f"Saved {X.shape} float32 array to {path}")


if __name__ == "__main__":
    main()
