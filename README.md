# SC4064_Project

## Directory Layout

- `scripts/`
  - `generate_dataset.py`
  - `generate_all_datasets.py`
  - `convert_npy_to_bin_soa.py`
  - `sbin_utils.py`
  - `sklearn_kmeans_from_bin.py`
  - `plot_2d_datasets.py`
  - `plot_results.py`
- `data_bin/` converted `.bin` files (SoA + header)
- `dist_assign_from_sbin.cu` CUDA entrypoint for direct SBIN loading
- existing CUDA files:
  - `dist_assign.cu`
  - `dist_assign_opt.cu`
  - `dist_assign_opt_cublas.cu`

## Python Environment

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## SBIN Format Specification

Binary file = **64-byte header** + **float32 SoA payload**

- Little-endian header fields:
  - `magic`: `SBIN` (4 bytes)
  - `version`: `uint32` (currently `1`)
  - `n`: `uint64` number of points
  - `d`: `uint32` dimensions
  - `k_meta`: `uint32` cluster count metadata (from filename if available)
  - `dtype_code`: `uint32` (`1` means `float32`)
  - `reserved`: 36 bytes padding
- Payload layout (SoA):
  - `x_dim0_all_points | x_dim1_all_points | ... | x_dim(d-1)_all_points`
  - each value is `float32`

## Reproduce Data and Conversion

### 1) (Optional) Generate `.npy` datasets inside repo

```bash
python3 scripts/generate_all_datasets.py
```

This writes to `data/` in repo root.

### 2) Convert all `.npy` in external data directory to repo `data_bin/`

```bash
python3 scripts/convert_npy_to_bin_soa.py \
  --input-dir /media/Pluto/lynu369/gpu_programming/data \
  --output-dir /media/Pluto/lynu369/gpu_programming/SC4064_Project/data_bin
```

### 3) (Recommended) Download `data_bin.zip` from Google Drive

If large datasets are not stored in git, use:

```bash
GDRIVE_FILE_ID="<your_file_id>" ./scripts/download_data_gdrive.sh
```

The script will:
- download `data_bin.zip` to repo root
- extract it to `data_bin/`
- auto-install `gdown` via `pip` if missing

## Run sklearn KMeans from `.bin`

Single file:

```bash
python3 scripts/sklearn_kmeans_from_bin.py \
  --bin data_bin/data_n5000_k10_d8.bin \
  --out results/sklearn_from_bin.csv
```

Whole directory:

```bash
python3 scripts/sklearn_kmeans_from_bin.py \
  --bin-dir data_bin \
  --out results/sklearn_from_bin.csv
```

## CUDA: Direct SoA Transfer Path

`dist_assign_from_sbin.cu` reads SBIN header/payload and uses `cudaMemcpyAsync` directly on SoA payload (no AoS transpose stage).

Compile:

```bash
nvcc -O3 dist_assign_from_sbin.cu -o dist_assign_from_sbin
```

Run:

```bash
./dist_assign_from_sbin data_bin/data_n5000_k10_d8.bin
```

Optional override K:

```bash
./dist_assign_from_sbin data_bin/data_n5000_k10_d8.bin 10
```

## Benchmark + Plot Tools

- benchmark from converted `.bin`:
  ```bash
  python3 scripts/sklearn_kmeans_from_bin.py \
    --bin-dir data_bin \
    --out results/sklearn_from_bin_all.csv
  ```
- plot benchmark from `.bin` results:
  ```bash
  python3 scripts/plot_results.py results/sklearn_from_bin_all.csv --out-dir results
  ```
- plot D=2 data:
  ```bash
  python3 scripts/plot_2d_datasets.py --data-dir data --out-dir results
  ```