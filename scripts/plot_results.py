#!/usr/bin/env python3
"""
Read sklearn benchmark CSV and produce line plots: time vs N, time vs K, time vs D.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_sweep_n(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[(df["k_clusters"] == 50) & (df["n_dims"] == 32)].sort_values("n_points")
    if sub.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(sub["n_points"], sub["total_time_s"], marker="o", linestyle="-")
    ax.set_xlabel("N (number of points)")
    ax.set_ylabel("Total time (s)")
    ax.set_title("sklearn K-means: time vs N (K=50, D=32)")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "time_vs_n.png", dpi=150)
    plt.close(fig)


def plot_sweep_k(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[(df["n_points"] == 1_000_000) & (df["n_dims"] == 32)].sort_values("k_clusters")
    if sub.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(sub["k_clusters"], sub["total_time_s"], marker="o", linestyle="-")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Total time (s)")
    ax.set_title("sklearn K-means: time vs K (N=1M, D=32)")
    fig.tight_layout()
    fig.savefig(out_dir / "time_vs_k.png", dpi=150)
    plt.close(fig)


def plot_sweep_d(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[(df["n_points"] == 1_000_000) & (df["k_clusters"] == 50)].sort_values("n_dims")
    if sub.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(sub["n_dims"], sub["total_time_s"], marker="o", linestyle="-")
    ax.set_xlabel("D (dimensions)")
    ax.set_ylabel("Total time (s)")
    ax.set_title("sklearn K-means: time vs D (N=1M, K=50)")
    fig.tight_layout()
    fig.savefig(out_dir / "time_vs_d.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sklearn benchmark results from CSV.")
    parser.add_argument("csv_path", type=str, nargs="?", default="sklearn_benchmark_results.csv")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save plot PNGs")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    for col in ["n_points", "k_clusters", "n_dims", "total_time_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    plot_sweep_n(df, out_dir)
    plot_sweep_k(df, out_dir)
    plot_sweep_d(df, out_dir)
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
