#!/bin/bash

DATA_DIR="data_bin"
MODE=$1

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory '$DATA_DIR' does not exist."
    exit 1
fi

# Ensure the directory contains at least one .bin file
shopt -s nullglob
SBIN_FILES=("$DATA_DIR"/*.bin)

if [ ${#SBIN_FILES[@]} -eq 0 ]; then
    echo "No .bin files found in $DATA_DIR."
    exit 1
fi

for file in "${SBIN_FILES[@]}"; do
    FILENAME=$(basename "$file")
    echo "=================================================="
    echo "Processing dataset: $FILENAME"
    echo "=================================================="

    if [ "$MODE" == "profile" ]; then
        echo "--> Profiling Baseline..."
        nsys profile --trace=cuda --stats=true --force-overwrite true -o "baseline_${FILENAME}" ./kmeans "$file"
        
        echo "--> Profiling Optimized Pipeline..."
        nsys profile --trace=cuda --stats=true --force-overwrite true -o "opt_${FILENAME}" ./kmeans_opt "$file"
    else
        echo "--> Running Baseline..."
        ./kmeans "$file"
        
        echo "--> Running Optimized Pipeline..."
        ./kmeans_opt "$file"
    fi
    echo ""
done