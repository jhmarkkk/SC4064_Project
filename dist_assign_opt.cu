#include "sbin.hh"

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cfloat>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

// CUDA Error Macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// Optimized CUDA Kernel: Chunked Execution on Global SoA + Shared Memory
__global__ void assignClustersKernel_Opt(
    const float* __restrict__ points_SoA, 
    const float* __restrict__ centroids, 
    int* __restrict__ assignments, 
    int chunk_size, int K, int D, int N, int offset) 
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

    // Boundary check using the dynamic chunk_size
    if (idx < chunk_size) {
        int global_idx = offset + idx;
        float min_dist = FLT_MAX;
        int best_cluster = -1;

        // 3. Calculate distance to all K centroids
        for (int k = 0; k < K; ++k) {
            float current_dist = 0.0f;

            for (int d = 0; d < D; ++d) {
                // Read from global SoA layout directly
                float pt_dim_val = points_SoA[d * N + global_idx];

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
        assignments[global_idx] = best_cluster;
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

    size_t total_points_bytes = static_cast<size_t>(N) * D * sizeof(float);
    size_t total_assign_bytes = static_cast<size_t>(N) * sizeof(int);
    size_t centroids_bytes = static_cast<size_t>(K) * D * sizeof(float);

    // 1. Allocate PINNED Host Memory for asynchronous transfers
    float *h_points_SoA, *h_centroids;
    int *h_assignments;
    CHECK_CUDA(cudaMallocHost(&h_points_SoA, total_points_bytes));
    CHECK_CUDA(cudaMallocHost(&h_centroids, centroids_bytes));
    CHECK_CUDA(cudaMallocHost(&h_assignments, total_assign_bytes));

    // 2. Initialize Data using SBIN loader
    if (!loadSbinSoA(argv[1], h_points_SoA, N, D)) {
        return 1;
    }

    // Initialize centroids using the first K points
    for(int k = 0; k < K; ++k) {
        for(int d = 0; d < D; ++d) {
            h_centroids[k * D + d] = h_points_SoA[d * N + k];
        }
    }

    // 3. Allocate Device Memory
    float *d_points_SoA, *d_centroids;
    int *d_assignments;
    CHECK_CUDA(cudaMalloc(&d_points_SoA, total_points_bytes));
    CHECK_CUDA(cudaMalloc(&d_centroids, centroids_bytes));
    CHECK_CUDA(cudaMalloc(&d_assignments, total_assign_bytes));

    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids, centroids_bytes, cudaMemcpyHostToDevice));

    // 4. Calculate generic kernel launch parameters
    int minGridSize, blockSize;
    size_t sharedMemSize = static_cast<size_t>(K) * D * sizeof(float);

    // We base the occupancy calculation on the largest chunk size
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, 
        assignClustersKernel_Opt, 
        sharedMemSize, N)); 

    std::cout << "--- Launch Configuration (Optimized, Chunked SoA) ---" << std::endl;
    std::cout << "N=" << N << " D=" << D << " K=" << K << std::endl;
    std::cout << "Block Size: " << blockSize << std::endl;

    // Create CUDA events for profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Sweeping number of streams configuration
    std::vector<int> test_streams = {1, 2, 4, 8, 16, 32};
    float best_time = FLT_MAX;
    int best_stream_count = -1;

    for (int num_streams : test_streams) {
        std::cout << "Testing NUM_STREAMS = " << num_streams << " ..." << std::endl;

        // 1. Calculate chunks and offsets for this iteration
        std::vector<int> stream_sizes(num_streams, 0);
        std::vector<size_t> stream_int_offsets(num_streams, 0);
        size_t current_int_offset = 0;
        int base_chunk = N / num_streams;
        int remainder = N % num_streams;

        for (int i = 0; i < num_streams; ++i) {
            stream_sizes[i] = base_chunk + (i < remainder ? 1 : 0);
            stream_int_offsets[i] = current_int_offset;
            current_int_offset += stream_sizes[i];
        }

        // 2. Create CUDA Streams
        std::vector<cudaStream_t> streams(num_streams);
        for (int i = 0; i < num_streams; ++i) {
            CHECK_CUDA(cudaStreamCreate(&streams[i]));
        }

        // 3. Start Profiling
        cudaEventRecord(start, 0);

        // 4. Asynchronous Execution Pipeline (Transfer -> Calculate -> Retrieve)
        for (int i = 0; i < num_streams; ++i) {
            int current_chunk_size = stream_sizes[i];

            if (current_chunk_size == 0)
                continue; 

            size_t int_offset = stream_int_offsets[i];
            size_t current_assign_bytes = static_cast<size_t>(current_chunk_size) * sizeof(int);

            // Grid parameters
            int gridSize = (current_chunk_size + blockSize - 1) / blockSize;

            // Step A: Async Copy strided SoA chunk from Host to Device in ONE API call
            size_t width_bytes = static_cast<size_t>(current_chunk_size) * sizeof(float);
            size_t pitch_bytes = static_cast<size_t>(N) * sizeof(float);

            CHECK_CUDA(cudaMemcpy2DAsync(
                d_points_SoA + int_offset, // Destination pointer (starts at chunk offset)
                pitch_bytes,               // Destination pitch (stride between dimensions)
                h_points_SoA + int_offset, // Source pointer (starts at chunk offset)
                pitch_bytes,               // Source pitch
                width_bytes,               // Contiguous bytes to copy per dimension
                D,                         // Number of dimensions (rows)
                cudaMemcpyHostToDevice, 
                streams[i]));

            // Step B: Execute Distance Kernel directly on the SoA data
            assignClustersKernel_Opt<<<gridSize, blockSize, sharedMemSize, streams[i]>>>(
                d_points_SoA, 
                d_centroids, 
                d_assignments, 
                current_chunk_size, K, D, N,
                static_cast<int>(int_offset));

            // Step C: Async Result Copy
            CHECK_CUDA(cudaMemcpyAsync(
                h_assignments + int_offset, 
                d_assignments + int_offset, 
                current_assign_bytes, cudaMemcpyDeviceToHost, streams[i]));
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        // Record stop event
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Calculate total execution time
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

        std::cout << "\n--- Performance Metrics (Optimized, Chunked SoA) ---" << std::endl;
        std::cout << "Total Points (N): " << N << std::endl;
        std::cout << "Dimensions (D): " << D << std::endl;
        std::cout << "Clusters (K): " << K << std::endl;
        std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
        std::cout << "Effective Bandwidth: " << effective_bandwidth_GBs << " GB/s" << std::endl;
        std::cout << "Throughput: " << throughput_GFLOPS << " GFLOPS" << std::endl;

        if (milliseconds < best_time) {
            best_time = milliseconds;
            best_stream_count = num_streams;
        }

        // 5. Cleanup streams for next loop
        for (int i = 0; i < num_streams; ++i) {
            CHECK_CUDA(cudaStreamDestroy(streams[i]));
        }
    }

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "N=" << N << " D=" << D << " K=" << K << std::endl;
    std::cout << "Block Size: " << blockSize << std::endl;
    std::cout << "🏆 Best NUM_STREAMS = " << best_stream_count 
              << " (" << best_time << " ms)" << std::endl;

    // 8. Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    CHECK_CUDA(cudaFree(d_points_SoA)); 
    CHECK_CUDA(cudaFree(d_centroids)); 
    CHECK_CUDA(cudaFree(d_assignments));
    CHECK_CUDA(cudaFreeHost(h_points_SoA)); 
    CHECK_CUDA(cudaFreeHost(h_centroids)); 
    CHECK_CUDA(cudaFreeHost(h_assignments));

    return 0;
}