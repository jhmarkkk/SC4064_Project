#!/usr/bin/env python3
"""
Convert .npy datasets (AoS: N x D float32) into SBIN (SoA + 64-byte header).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sbin_utils import parse_nkd_from_name, write_sbin


def convert_file(src: Path, out_dir: Path, overwrite: bool = False) -> Path:
    arr = np.load(src, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"{src} is not 2D: shape={arr.shape}")
    n, d = int(arr.shape[0]), int(arr.shape[1])
    n_name, k_name, d_name = parse_nkd_from_name(src)
    if d_name is not None and d_name != d:
        raise ValueError(f"{src} has name d={d_name} but array d={d}")
    k_meta = int(k_name or 0)

    out_path = out_dir / (src.stem + ".bin")
    if out_path.exists() and not overwrite:
        print(f"Skip (exists): {out_path.name}")
        return out_path

    # SoA layout: (D, N), row-major contiguous payload
    soa = np.ascontiguousarray(np.asarray(arr, dtype=np.float32).T)
    write_sbin(out_path, soa_data=soa, n=n, d=d, k_meta=k_meta)
    print(f"Converted: {src.name} -> {out_path.name}  shape=({n},{d}) k_meta={k_meta}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .npy datasets to SBIN (.bin) SoA format.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing .npy files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save .bin files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .bin files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    for src in npy_files:
        convert_file(src, output_dir, overwrite=args.overwrite)

    print(f"Done. Converted {len(npy_files)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()
