#!/usr/bin/env python3
"""
Run sklearn KMeans directly from SBIN (.bin) datasets.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import sklearn
from sklearn.cluster import KMeans

from sbin_utils import load_sbin_to_aos, parse_nkd_from_name

CSV_COLUMNS = [
    "dataset_file",
    "n_points",
    "k_clusters",
    "n_dims",
    "total_time_s",
    "time_per_iter_s",
    "n_iters",
    "final_inertia",
    "sklearn_version",
    "timestamp",
]


def run_one(bin_path: Path, k: int | None, max_iter: int, seed: int) -> dict:
    X, meta = load_sbin_to_aos(bin_path)
    n = int(meta["n"])
    d = int(meta["d"])
    _, k_name, _ = parse_nkd_from_name(bin_path)
    k_use = int(k or k_name or meta.get("k_meta", 0) or 8)
    if k_use <= 0:
        raise ValueError(f"Invalid K for {bin_path}, got {k_use}")

    model = KMeans(
        n_clusters=k_use,
        n_init=1,
        max_iter=max_iter,
        algorithm="lloyd",
        init="random",
        random_state=seed,
    )
    t0 = time.perf_counter()
    model.fit(X)
    t1 = time.perf_counter()
    total_time_s = t1 - t0
    n_iters = int(model.n_iter_)
    return {
        "dataset_file": bin_path.name,
        "n_points": n,
        "k_clusters": k_use,
        "n_dims": d,
        "total_time_s": total_time_s,
        "time_per_iter_s": total_time_s / n_iters if n_iters else 0.0,
        "n_iters": n_iters,
        "final_inertia": float(model.inertia_),
        "sklearn_version": sklearn.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sklearn KMeans from SBIN files.")
    parser.add_argument("--bin", type=str, help="Single .bin file path")
    parser.add_argument("--bin-dir", type=str, help="Directory containing .bin files")
    parser.add_argument("--k", type=int, default=0, help="Override K; default from file name/header")
    parser.add_argument("--max-iter", type=int, default=20, help="KMeans max_iter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="sklearn_benchmark_from_bin.csv", help="Output CSV path")
    args = parser.parse_args()

    files: list[Path] = []
    if args.bin:
        files.append(Path(args.bin))
    if args.bin_dir:
        files.extend(sorted(Path(args.bin_dir).glob("*.bin")))
    files = [f for f in files if f.exists()]
    if not files:
        raise ValueError("Provide --bin or --bin-dir with existing .bin files.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for p in files:
            row = run_one(p, args.k or None, args.max_iter, args.seed)
            writer.writerow(row)
            print(
                f"{p.name}: N={row['n_points']:,} K={row['k_clusters']} D={row['n_dims']} "
                f"time={row['total_time_s']:.2f}s iters={row['n_iters']}"
            )

    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
