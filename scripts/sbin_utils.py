#!/usr/bin/env python3
"""
Utilities for SBIN (SoA binary) format.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

MAGIC = b"SBIN"
VERSION = 1
DTYPE_FLOAT32 = 1
HEADER_SIZE = 64
_HEADER_STRUCT = struct.Struct("<4sI Q I I I 36s")


def parse_nkd_from_name(path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    match = re.search(r"n(\d+)_k(\d+)_d(\d+)", path.stem)
    if not match:
        return None, None, None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def write_sbin(path: Path, soa_data: np.ndarray, n: int, d: int, k_meta: int = 0) -> None:
    if soa_data.dtype != np.float32:
        raise ValueError("soa_data must be float32")
    if soa_data.ndim != 2 or soa_data.shape != (d, n):
        raise ValueError(f"soa_data shape must be (d, n)=({d}, {n}), got {soa_data.shape}")

    header = _HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        n,
        d,
        k_meta,
        DTYPE_FLOAT32,
        b"\x00" * 36,
    )
    with path.open("wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(soa_data).reshape(-1).tobytes(order="C"))


def read_sbin_header(path: Path) -> dict:
    with path.open("rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise ValueError(f"Invalid SBIN header size in {path}")
    magic, version, n, d, k_meta, dtype_code, _ = _HEADER_STRUCT.unpack(raw)
    if magic != MAGIC:
        raise ValueError(f"Invalid magic in {path}: {magic!r}")
    if version != VERSION:
        raise ValueError(f"Unsupported version {version} in {path}")
    if dtype_code != DTYPE_FLOAT32:
        raise ValueError(f"Unsupported dtype_code {dtype_code} in {path}")
    return {"version": version, "n": n, "d": d, "k_meta": k_meta, "dtype_code": dtype_code}


def load_sbin_to_soa(path: Path, mmap_mode: str = "r") -> tuple[np.ndarray, dict]:
    meta = read_sbin_header(path)
    n = int(meta["n"])
    d = int(meta["d"])
    payload = np.memmap(path, dtype=np.float32, mode=mmap_mode, offset=HEADER_SIZE, shape=(d, n), order="C")
    return payload, meta


def load_sbin_to_aos(path: Path, mmap_mode: str = "r") -> tuple[np.ndarray, dict]:
    soa, meta = load_sbin_to_soa(path, mmap_mode=mmap_mode)
    aos = np.asarray(soa.T, dtype=np.float32)
    return aos, meta
