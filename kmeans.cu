#include "sbin.hh"

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

// =============================================================================
// Host helper: divide accumulated sums by counts → new centroid positions
// =============================================================================
void computeNewCentroids(const float* centroid_sums,
                         const int*   centroid_counts,
                               float* centroids,
                         int K, int D)
{
    for (int k = 0; k < K; ++k) {
        int count = centroid_counts[k];
        for (int d = 0; d < D; ++d)
            centroids[k * D + d] = (count > 0)
                                   ? centroid_sums[k * D + d] / (float)count
                                   : 0.0f;
    }
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

    int max_iters = 100;
    if (argc >= 5) max_iters = std::stoi(argv[4]);

    size_t points_size      = static_cast<size_t>(N) * D * sizeof(float);
    size_t centroids_size   = static_cast<size_t>(K) * D * sizeof(float);
    size_t assignments_size = static_cast<size_t>(N) * sizeof(int);

    // ── Host allocations ──
    std::vector<float> h_points(static_cast<size_t>(N) * D);
    std::vector<float> h_centroids(static_cast<size_t>(K) * D);
    std::vector<int>   h_assignments(N);
    std::vector<float> h_centroid_sums(static_cast<size_t>(K) * D);
    std::vector<int>   h_centroid_counts(K);

    if (!loadSbinSoA(argv[1], h_points.data(), N, D)) return 1;

    // ── Initialise centroids: pick K random points ──
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distrib(0, N - 1);

    std::cout << "Initializing centroids using random seed: " << seed << std::endl;
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

    std::cout << "--- Launch Configuration ---" << std::endl;
    std::cout << "N=" << N << "  D=" << D << "  K=" << K
              << "  max_iters=" << max_iters << std::endl;
    std::cout << "Block size           : " << blockSize << std::endl;
    std::cout << "Theoretical occupancy: " << std::fixed << std::setprecision(2)
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

        // Step 4: copy sums and counts to host, divide to get new centroids
        CHECK_CUDA(cudaMemcpy(h_centroid_sums.data(),   d_centroid_sums,
                              centroids_size,   cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_centroid_counts.data(), d_centroid_counts,
                              K * sizeof(int),  cudaMemcpyDeviceToHost));

        computeNewCentroids(h_centroid_sums.data(), h_centroid_counts.data(),
                            h_centroids.data(), K, D);

        // Step 5: upload updated centroids back to device
        CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(),
                              centroids_size, cudaMemcpyHostToDevice));
    }

    // Copy final assignments back
    CHECK_CUDA(cudaMemcpy(h_assignments.data(), d_assignments,
                          assignments_size, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;

    // ── Performance metrics ──
    double total_bytes       = (double)N * (8.0 * D + 8.0);
    double effective_bw_GBs  = (total_bytes / 1e9) / seconds;
    double total_flops       = 3.0 * (double)N * K * D * max_iters;
    double throughput_GFLOPS = (total_flops / 1e9) / seconds;

    std::cout << "\n--- Performance Metrics ---" << std::endl;
    std::cout << "Total points (N)    : " << N << std::endl;
    std::cout << "Dimensions (D)      : " << D << std::endl;
    std::cout << "Clusters (K)        : " << K << std::endl;
    std::cout << "Iterations          : " << max_iters << std::endl;
    std::cout << "Total time          : " << milliseconds << " ms" << std::endl;
    std::cout << "Time per iteration  : " << milliseconds / max_iters << " ms" << std::endl;
    std::cout << "Effective bandwidth : " << effective_bw_GBs << " GB/s" << std::endl;
    std::cout << "Throughput          : " << throughput_GFLOPS << " GFLOPS" << std::endl;

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
