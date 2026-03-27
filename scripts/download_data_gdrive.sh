#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/download_data_gdrive.sh
#
# Before first use:
# 1) Upload data_bin.zip to Google Drive.
# 2) Set GDRIVE_FILE_ID below (or pass via env var).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/data_bin"
ZIP_PATH="${REPO_ROOT}/data_bin.zip"

# Replace with your shared Google Drive file ID (for data_bin.zip).
GDRIVE_FILE_ID="${GDRIVE_FILE_ID:-REPLACE_WITH_FILE_ID}"

if ! command -v gdown >/dev/null 2>&1; then
  echo "[INFO] gdown not found. Installing via pip..."
  python3 -m pip install --user gdown
fi

if [[ "${GDRIVE_FILE_ID}" == "REPLACE_WITH_FILE_ID" ]]; then
  echo "[ERROR] Please set GDRIVE_FILE_ID first."
  echo "Example:"
  echo "  GDRIVE_FILE_ID='1AbCdEf...' ./scripts/download_data_gdrive.sh"
  exit 1
fi

echo "[INFO] Downloading data_bin.zip from Google Drive file id: ${GDRIVE_FILE_ID}"
gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O "${ZIP_PATH}"

echo "[INFO] Extracting ${ZIP_PATH} ..."
rm -rf "${OUT_DIR}"
unzip -o "${ZIP_PATH}" -d "${REPO_ROOT}"

echo "[INFO] Download and extract complete. Files in ${OUT_DIR}:"
ls -lh "${OUT_DIR}"/*.bin
