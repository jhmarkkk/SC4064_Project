#include "sbin.hh"
#include "common_kernels.hh"
#include "save_results.hh"

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cfloat>
#include <vector>
#include <random>
#include <iomanip>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// =============================================================================
// Kernel 1: Cluster Assignment
// Each thread handles one point and finds its nearest centroid.
// Centroids are loaded into shared memory once per block to avoid redundant
// global memory reads across threads.
// =============================================================================
__global__ void assignClustersKernel(const float* __restrict__ points,
                                     const float* __restrict__ centroids,
                                     int*   __restrict__ assignments,
                                     int N, int K, int D)
{
    extern __shared__ float s_centroids[];

    // Collaboratively load all centroids into shared memory
    int total_centroid_elements = K * D;
    for (int i = threadIdx.x; i < total_centroid_elements; i += blockDim.x)
        s_centroids[i] = centroids[i];

    __syncthreads();

    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= N) return;

    float min_dist    = FLT_MAX;
    int   best_cluster = -1;

    for (int k = 0; k < K; ++k) {
        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = points[d * N + point_idx] - s_centroids[k * D + d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist     = dist;
            best_cluster = k;
        }
    }

    assignments[point_idx] = best_cluster;
}

// =============================================================================
// Kernel 2: Naive Atomic Centroid Recalculation
// Each thread processes one point and atomically adds its coordinates into the
// running sum for its assigned cluster, and increments that cluster's count.
// Called "naive" because every thread issues D atomic operations directly to
// global memory — causes contention when many points share the same cluster.
// The division (sum / count) is done on the host after this kernel returns.
// =============================================================================
__global__ void recalcCentroidsKernel(const float* __restrict__ points,
                                      const int*   __restrict__ assignments,
                                            float* __restrict__ centroid_sums,
                                            int*   __restrict__ centroid_counts,
                                      int N, int K, int D)
{
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= N) return;

    int cluster = assignments[point_idx];

    for (int d = 0; d < D; ++d)
        atomicAdd(&centroid_sums[cluster * D + d], points[d * N + point_idx]);

    atomicAdd(&centroid_counts[cluster], 1);
}

// Kernel 3: Divide sums by counts entirely on GPU — no host roundtrip needed
__global__ void updateCentroidsKernel(const float* __restrict__ centroid_sums,
                                      const int*   __restrict__ centroid_counts,
                                            float* __restrict__ centroids,
                                      int K, int D)
{
    // Each thread handles one (cluster, dimension) pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * D) return;

    int k = idx / D;  // which cluster
    int d = idx % D;  // which dimension

    int count = centroid_counts[k];
    centroids[idx] = (count > 0)
                     ? centroid_sums[idx] / (float)count
                     : 0.0f;
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_bin_file> [K_override] [Seed] [max_iters]\n";
        return 1;
    }

    // ── Load SBIN header ──
    SbinHeader header{};
    {
        std::ifstream in(argv[1], std::ios::binary);
        if (!in) {
            std::cerr << "Failed to open SBIN file: " << argv[1] << std::endl;
            return 1;
        }
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!in || std::memcmp(header.magic, "SBIN", 4) != 0) {
            std::cerr << "Invalid SBIN header.\n";
            return 1;
        }
    }

    int N = static_cast<int>(header.n);
    int D = static_cast<int>(header.d);
    int K = static_cast<int>(header.k_meta > 0 ? header.k_meta : 10);
    if (argc >= 3) K = std::stoi(argv[2]);

    int seed = 42;
    if (argc >= 4) seed = std::stoi(argv[3]);

    int max_iters = 20;
    if (argc >= 5) max_iters = std::stoi(argv[4]);

    size_t points_size      = static_cast<size_t>(N) * D * sizeof(float);
    size_t centroids_size   = static_cast<size_t>(K) * D * sizeof(float);
    size_t assignments_size = static_cast<size_t>(N) * sizeof(int);

    // ── Host allocations ──
    std::vector<float> h_points(static_cast<size_t>(N) * D);
    std::vector<float> h_centroids(static_cast<size_t>(K) * D);
    std::vector<int>   h_assignments(N);

    if (!loadSbinSoA(argv[1], h_points.data(), N, D)) return 1;

    // ── Initialise centroids: pick K random points ──
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distrib(0, N - 1);

    std::cerr << "Initializing centroids using random seed: " << seed << std::endl;
    for (int k = 0; k < K; ++k) {
        int idx = distrib(gen);
        for (int d = 0; d < D; ++d)
            h_centroids[k * D + d] = h_points[d * N + idx];
    }

    // ── Device allocations ──
    float *d_points, *d_centroids, *d_centroid_sums;
    int   *d_assignments, *d_centroid_counts;

    CHECK_CUDA(cudaMalloc(&d_points,         points_size));
    CHECK_CUDA(cudaMalloc(&d_centroids,       centroids_size));
    CHECK_CUDA(cudaMalloc(&d_assignments,     assignments_size));
    CHECK_CUDA(cudaMalloc(&d_centroid_sums,   centroids_size));
    CHECK_CUDA(cudaMalloc(&d_centroid_counts, K * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_points, h_points.data(), points_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(), centroids_size, cudaMemcpyHostToDevice));

    // ── Shared memory for assignment kernel ──
    size_t sharedMemSize = static_cast<size_t>(K) * D * sizeof(float);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (sharedMemSize > prop.sharedMemPerBlockOptin) {
        std::cerr << "[ERROR] Required shared memory (" << sharedMemSize
                  << " bytes) exceeds hardware limit ("
                  << prop.sharedMemPerBlockOptin << " bytes)." << std::endl;
        return 1;
    }

    CHECK_CUDA(cudaFuncSetAttribute(assignClustersKernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    sharedMemSize));

    CHECK_CUDA(cudaFuncSetAttribute(computeInertiaKernel, 
                                cudaFuncAttributeMaxDynamicSharedMemorySize, 
                                sharedMemSize));

    int minGridSize, blockSize = 0;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, assignClustersKernel, sharedMemSize, N));

    if (blockSize == 0) {
        std::cerr << "[ERROR] Could not determine a valid block size." << std::endl;
        return 1;
    }

    int gridSize = (N + blockSize - 1) / blockSize;

    int numBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM, assignClustersKernel, blockSize, sharedMemSize);
    float occupancy = (numBlocksPerSM * blockSize) / (float)prop.maxThreadsPerMultiProcessor;

    std::cerr << "--- Launch Configuration ---" << std::endl;
    std::cerr << "N=" << N << "  D=" << D << "  K=" << K
              << "  max_iters=" << max_iters << std::endl;
    std::cerr << "Block size           : " << blockSize << std::endl;
    std::cerr << "Theoretical occupancy: " << std::fixed << std::setprecision(2)
              << (occupancy * 100.0f) << "%" << std::endl;

    // ── CUDA events for timing the full clustering loop ──
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // ── K-means iteration loop ──
    for (int iter = 0; iter < max_iters; ++iter) {

        // Step 1: assign each point to its nearest centroid
        assignClustersKernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_points, d_centroids, d_assignments, N, K, D);

        // Step 2: zero accumulators
        CHECK_CUDA(cudaMemset(d_centroid_sums,   0, centroids_size));
        CHECK_CUDA(cudaMemset(d_centroid_counts, 0, K * sizeof(int)));

        // Step 3: accumulate sums and counts atomically
        recalcCentroidsKernel<<<gridSize, blockSize>>>(
            d_points, d_assignments,
            d_centroid_sums, d_centroid_counts,
            N, K, D);

        // --- Step 4: Updating Centroid Kernel ---
        int kd_total  = K * D;
        int kd_blocks = (kd_total + blockSize - 1) / blockSize;
        updateCentroidsKernel<<<kd_blocks, blockSize>>>(
            d_centroid_sums, d_centroid_counts, d_centroids, K, D);
    }

    // Copy final assignments back
    CHECK_CUDA(cudaMemcpy(h_assignments.data(), d_assignments,
                          assignments_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_centroids.data(), d_centroids,
                      centroids_size, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // ── Save results ──
    // Derive base name from input file e.g. data_bin/foo.bin → foo
    std::string input_path = argv[1];
    std::string base = input_path.substr(input_path.find_last_of("/\\") + 1);
    base = base.substr(0, base.find_last_of('.'));
    std::string tag    = "baseline";
    std::string outdir = "baseline_results";
    // Create output directory if it doesn't exist
    system(("mkdir -p " + outdir).c_str());
    std::string prefix = outdir + "/" + base + "_" + tag;
    saveAssignmentsCSV(h_assignments,     prefix + "_assignments.csv");
    saveCentroidsCSV  (h_centroids, K, D, prefix + "_centroids.csv");

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;

    // ── Performance metrics ──
    // Kernel 1 — assignment:
    // Reads: N×D (points) + K×D (centroids)
    // Writes: N (assignments)
    // Total: N*D*4 + K*D*4 + N*4
    // Kernel 2 — recalc:
    // Reads: N×D (points) + N (assignments)
    // Writes: K×D (sums) + K (counts)
    // Total: N*D*4 + N*4 + K*D*4 + K*4
    // Kernel 3 — update:
    // Reads: K×D (sums) + K (counts)
    // Writes: K×D (centroids)
    // Total: K*D*4 + K*4 + K*D*4
    double bytes_assign = (double)N * D * 4 + (double)K * D * 4 + (double)N * 4;
    double bytes_recalc = (double)N * D * 4 + (double)N * 4 + (double)K * D * 4 + (double)K * 4;
    double bytes_update = (double)K * D * 4 + (double)K * 4 + (double)K * D * 4;
    double total_bytes  = (bytes_assign + bytes_recalc + bytes_update) * max_iters;
    double effective_bw_GBs  = (total_bytes / 1e9) / seconds;

    // Kernel 1 — assignment:
    // Per point per cluster per dimension: 1 subtract, 1 multiply, 1 add = 3 FLOPs
    // Total: N * K * D * 3
    // Kernel 2 — recalc:
    // Per point per dimension: 1 atomicAdd = 1 FLOP
    // Per point: 1 atomicAdd for count = 1 FLOP
    // Total: N * D * 1 + N * 1 = N * (D + 1)
    // Kernel 3 — update:
    // Per (cluster, dimension): 1 divide = 1 FLOP
    // Total: K * D * 1
    double flops_assign = 3.0 * (double)N * K * D;
    double flops_recalc = (double)N * (D + 1);
    double flops_update = (double)K * D;
    double total_flops  = (flops_assign + flops_recalc + flops_update) * max_iters;
    double throughput_GFLOPS = (total_flops / 1e9) / seconds;

    std::cerr << "\n--- Performance Metrics ---" << std::endl;
    std::cerr << "Total points (N)    : " << N << std::endl;
    std::cerr << "Dimensions (D)      : " << D << std::endl;
    std::cerr << "Clusters (K)        : " << K << std::endl;
    std::cerr << "Iterations          : " << max_iters << std::endl;
    std::cerr << "Total time          : " << milliseconds << " ms" << std::endl;
    std::cerr << "Time per iteration  : " << milliseconds / max_iters << " ms" << std::endl;
    std::cerr << "Effective bandwidth : " << effective_bw_GBs << " GB/s" << std::endl;
    std::cerr << "Throughput          : " << throughput_GFLOPS << " GFLOPS" << std::endl;

    // Inertia computation
    double* d_inertia;
    CHECK_CUDA(cudaMalloc(&d_inertia, sizeof(double)));
    CHECK_CUDA(cudaMemset(d_inertia, 0, sizeof(double)));
    computeInertiaKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_points, d_centroids, d_assignments,
        d_inertia, N, K, D);

    double h_inertia = 0.0;
    CHECK_CUDA(cudaMemcpy(&h_inertia, d_inertia, sizeof(double),
                        cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_inertia));
    std::cerr << "Final inertia: " << std::fixed << std::setprecision(2)
            << h_inertia << std::endl;

    // cout into csv files
    std::cout << base << ","
          << N << ","
          << K << ","
          << D << ","
          << std::fixed << std::setprecision(6) << seconds << ","
          << seconds / max_iters << ","
          << max_iters << ","
          << std::setprecision(2) << h_inertia << ","
          << "baseline"
          << std::endl;

    // ── Cleanup ──
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK_CUDA(cudaFree(d_points));
    CHECK_CUDA(cudaFree(d_centroids));
    CHECK_CUDA(cudaFree(d_assignments));
    CHECK_CUDA(cudaFree(d_centroid_sums));
    CHECK_CUDA(cudaFree(d_centroid_counts));

    return 0;
}
