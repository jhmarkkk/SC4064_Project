# ─────────────────────────────────────────────────────────────────────────────
# K-Means GPU Optimization Pipeline
# ─────────────────────────────────────────────────────────────────────────────

NVCC        := nvcc
CXX_STD     := -std=c++17

# Set for ASPIRE 2A NVIDIA A100 GPUs
ARCH        := -gencode arch=compute_80,code=sm_80

# Optimization and debug flags (-lineinfo allows Nsight to map back to source lines)
OPT_FLAGS   := -O3 -lineinfo

# Warning flags
WARN_FLAGS  := -Xcompiler -Wall,-Wextra

NVCC_FLAGS  := $(CXX_STD) $(ARCH) $(OPT_FLAGS) $(WARN_FLAGS)

TARGETS     := kmeans kmeans_opt

.PHONY: all run profile clean

all: $(TARGETS)

# Build baseline
kmeans: kmeans.cu sbin.cc
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# Build optimized version
kmeans_opt: kmeans_opt.cu sbin.cc
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# Execute standard run script
run: all
	bash run_all.sh run

# Execute profiling run script
profile: all
	bash run_all.sh profile

# Clean up binaries and Nsight profiling artifacts
clean:
	rm -f $(TARGETS) *.nsys-rep *.sqlite *.qdrep