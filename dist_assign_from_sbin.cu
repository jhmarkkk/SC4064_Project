#include <cuda_runtime.h>
#include <iostream>
#include <cfloat>
#include <vector>
#include <random>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <iomanip>

// CUDA Error Macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

struct SbinHeader {
    char magic[4];
    uint32_t version;
    uint64_t n;
    uint32_t d;
    uint32_t k_meta;
    uint32_t dtype_code;
    char reserved[36];
};

static bool loadSbinSoA(const std::string& path, float* h_points_SoA, int N, int D) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open SBIN file: " << path << std::endl;
        return false;
    }

    SbinHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in) {
        std::cerr << "Failed to read SBIN header.\n";
        return false;
    }
    if (std::memcmp(header.magic, "SBIN", 4) != 0 || header.version != 1 || header.dtype_code != 1) {
        std::cerr << "Unsupported SBIN header.\n";
        return false;
    }
    if (static_cast<int>(header.n) != N || static_cast<int>(header.d) != D) {
        std::cerr << "Header mismatch. Expected N=" << N << " D=" << D
                  << " but got N=" << header.n << " D=" << header.d << std::endl;
        return false;
    }

    in.read(reinterpret_cast<char*>(h_points_SoA), static_cast<size_t>(N) * D * sizeof(float));
    if (!in) {
        std::cerr << "Failed to read SBIN payload.\n";
        return false;
    }
    return true;
}

// Optimized CUDA Kernel: Chunked SoA + Shared Memory
__global__ void assignClustersKernel_Opt(
    const float* __restrict__ points_soa_all,
    const float* __restrict__ centroids,
    int* __restrict__ assignments_chunk,
    int point_offset, int chunk_size, int N_total, int K, int D)
{
    // Dynamically allocated shared memory for centroids
    extern __shared__ float s_centroids[];

    // 1. Collaboratively load centroids into Shared Memory
    int total_centroid_elements = K * D;
    for (int i = threadIdx.x; i < total_centroid_elements; i += blockDim.x) {
        s_centroids[i] = centroids[i];
    }
    __syncthreads();

    // 2. Global thread index maps to a specific data point WITHIN this chunk
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunk_size) {
        int global_point_idx = point_offset + idx;
        float min_dist = FLT_MAX;
        int best_cluster = -1;

        // 3. Calculate distance to all K centroids
        for (int k = 0; k < K; ++k) {
            float current_dist = 0.0f;
            for (int d = 0; d < D; ++d) {
                // Read from SoA layout directly loaded from SBIN payload
                float pt_dim_val = points_soa_all[d * N_total + global_point_idx];
                // Read centroid from fast shared memory
                float diff = pt_dim_val - s_centroids[k * D + d];
                current_dist += diff * diff;
            }
            if (current_dist < min_dist) {
                min_dist = current_dist;
                best_cluster = k;
            }
        }
        // 4. Write the result
        assignments_chunk[idx] = best_cluster;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_bin_file> [K_override]\n";
        return 1;
    }

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
    if (argc >= 3) {
        K = std::stoi(argv[2]);
    }

    // TODO: Stream configuration
    const int NUM_STREAMS = 4;
    // Arrays to hold dynamic sizes and offsets for each stream
    std::vector<int> stream_sizes(NUM_STREAMS, 0);
    std::vector<size_t> stream_int_offsets(NUM_STREAMS, 0);

    size_t current_int_offset = 0;
    // Distribute N points across streams, accounting for remainder
    int base_chunk = N / NUM_STREAMS;
    int remainder = N % NUM_STREAMS;
    for (int i = 0; i < NUM_STREAMS; ++i) {
        // First 'remainder' streams get one extra point
        stream_sizes[i] = base_chunk + (i < remainder ? 1 : 0);
        // Record starting offsets for assignment pointers
        stream_int_offsets[i] = current_int_offset;
        // Update cumulative offsets
        current_int_offset += stream_sizes[i];
    }

    size_t total_points_bytes = static_cast<size_t>(N) * D * sizeof(float);
    size_t total_assign_bytes = static_cast<size_t>(N) * sizeof(int);
    size_t centroids_bytes = static_cast<size_t>(K) * D * sizeof(float);

    // 1. Allocate PINNED Host Memory for asynchronous transfers
    float *h_points_SoA, *h_centroids;
    int *h_assignments;
    CHECK_CUDA(cudaMallocHost(&h_points_SoA, total_points_bytes));
    CHECK_CUDA(cudaMallocHost(&h_centroids, centroids_bytes));
    CHECK_CUDA(cudaMallocHost(&h_assignments, total_assign_bytes));

    // 2. Load data directly from SBIN payload in SoA layout
    if (!loadSbinSoA(argv[1], h_points_SoA, N, D)) {
        return 1;
    }
    for (int i = 0; i < K * D; ++i) {
        h_centroids[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 3. Allocate Device Memory
    float *d_points_SoA, *d_centroids;
    int *d_assignments;
    CHECK_CUDA(cudaMalloc(&d_points_SoA, total_points_bytes));
    CHECK_CUDA(cudaMalloc(&d_centroids, centroids_bytes));
    CHECK_CUDA(cudaMalloc(&d_assignments, total_assign_bytes));

    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids, centroids_bytes, cudaMemcpyHostToDevice));

    // 4. Create CUDA Streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // 5. Dynamic Kernel Launch Configuration
    int minGridSize, blockSize;
    size_t sharedMemSize = static_cast<size_t>(K) * D * sizeof(float);
    // We base the occupancy calculation on the largest chunk size
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize,
        assignClustersKernel_Opt,
        sharedMemSize, stream_sizes[0]));

    // Calculate Occupancy
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int numBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, assignClustersKernel_Opt, blockSize, sharedMemSize);
    float occupancy = (numBlocksPerSM * blockSize) / (float)prop.maxThreadsPerMultiProcessor;

    std::cout << "--- Launch Configuration (Optimized, direct SBIN SoA input) ---" << std::endl;
    std::cout << "N=" << N << " D=" << D << " K=" << K << std::endl;
    std::cout << "Block Size: " << blockSize << std::endl;
    std::cout << "Theoretical Occupancy: " << std::fixed << std::setprecision(2) << (occupancy * 100.0f) << "%" << std::endl;

    // Create CUDA events for profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record start event on the default stream
    cudaEventRecord(start, 0);

    // Step A: Async copy full SoA payload from Host to Device
    CHECK_CUDA(cudaMemcpyAsync(
        d_points_SoA,
        h_points_SoA,
        total_points_bytes,
        cudaMemcpyHostToDevice,
        0));

    // 7. Asynchronous Execution Pipeline (Calculate -> Retrieve)
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int current_chunk_size = stream_sizes[i];
        if (current_chunk_size == 0) {
            continue;
        }

        size_t int_offset = stream_int_offsets[i];
        size_t current_assign_bytes = static_cast<size_t>(current_chunk_size) * sizeof(int);
        // Grid parameters
        int gridSize = (current_chunk_size + blockSize - 1) / blockSize;

        // Step B: Execute Distance Kernel on SoA data (no transpose stage)
        assignClustersKernel_Opt<<<gridSize, blockSize, sharedMemSize, streams[i]>>>(
            d_points_SoA,
            d_centroids,
            d_assignments + int_offset,
            static_cast<int>(int_offset),
            current_chunk_size,
            N,
            K,
            D);

        // Step C: Async Result Copy
        CHECK_CUDA(cudaMemcpyAsync(
            h_assignments + int_offset,
            d_assignments + int_offset,
            current_assign_bytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;

    // Calculate effective bandwidth:
    // Host to Device: N * D * 4 bytes
    // Kernel Read (Points): N * D * 4 bytes
    // Kernel Write (Assignments): N * 4 bytes
    // Device to Host: N * 4 bytes
    // Total Bytes = N * (8D + 8) bytes
    double total_bytes = (double)N * (8.0 * D + 8.0);
    double effective_bandwidth_GBs = (total_bytes / 1e9) / seconds;
    // Calculate throughput in GFLOPS:
    // a subtraction, a multiplication, and an addition for every dimension
    double total_flops = 3.0 * (double)N * K * D;
    double throughput_GFLOPS = (total_flops / 1e9) / seconds;

    std::cout << "\n--- Performance Metrics (Optimized, direct SBIN SoA input) ---" << std::endl;
    std::cout << "Total Points (N): " << N << std::endl;
    std::cout << "Dimensions (D): " << D << std::endl;
    std::cout << "Clusters (K): " << K << std::endl;
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << effective_bandwidth_GBs << " GB/s" << std::endl;
    std::cout << "Throughput: " << throughput_GFLOPS << " GFLOPS" << std::endl;

    // 8. Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaFree(d_points_SoA));
    CHECK_CUDA(cudaFree(d_centroids));
    CHECK_CUDA(cudaFree(d_assignments));
    CHECK_CUDA(cudaFreeHost(h_points_SoA));
    CHECK_CUDA(cudaFreeHost(h_centroids));
    CHECK_CUDA(cudaFreeHost(h_assignments));

    return 0;
}
