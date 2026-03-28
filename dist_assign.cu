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

// CUDA Error Macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// CUDA Kernel: Euclidean Distance Calculation & Cluster Assignment
__global__ void assignClustersKernel(const float* __restrict__ points, 
                                     const float* __restrict__ centroids, 
                                     int* __restrict__ assignments, 
                                     int N, int K, int D) {

    // Dynamically allocated shared memory for centroids
    // This addresses the "Memory Hierarchy" optimization goal
    extern __shared__ float s_centroids[];

    // 1. Collaboratively load centroids from Global to Shared Memory
    // Every thread in the block helps pull the centroid data into shared memory
    int total_centroid_elements = K * D;
    for (int i = threadIdx.x; i < total_centroid_elements; i += blockDim.x) {
        s_centroids[i] = centroids[i];
    }

    // Wait for all threads in the block to finish loading before proceeding
    __syncthreads(); 

    // 2. Map this specific thread to a specific data point
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check (in case N is not perfectly divisible by block size)
    if (point_idx < N) {
        float min_dist = FLT_MAX;
        int best_cluster = -1;

        // 3. Calculate Euclidean distance to all K centroids
        for (int k = 0; k < K; ++k) {
            float current_dist = 0.0f;

            for (int d = 0; d < D; ++d) {
                // Access point from global memory, centroid from fast shared memory
                float diff = points[d * N + point_idx] - s_centroids[k * D + d];
                current_dist += diff * diff; // Squared distance
            }

            // Track the minimum distance
            if (current_dist < min_dist) {
                min_dist = current_dist;
                best_cluster = k;
            }
        }

        // 4. Assign the point to the closest cluster's index
        assignments[point_idx] = best_cluster;
    }
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_bin_file> [K_override] [Seed]\n";
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

    // Parse Seed Argument
    int seed = 42; 
    if (argc >= 4) {
        seed = std::stoi(argv[3]);
    }

    size_t points_size = static_cast<size_t>(N) * D * sizeof(float);
    size_t centroids_size = static_cast<size_t>(K) * D * sizeof(float);
    size_t assignments_size = static_cast<size_t>(N) * sizeof(int);

    // Host memory allocation
    std::vector<float> h_points(static_cast<size_t>(N) * D);
    std::vector<float> h_centroids(static_cast<size_t>(K) * D);
    std::vector<int> h_assignments(N);

    // Load actual SBIN loading logic
    if (!loadSbinSoA(argv[1], h_points.data(), N, D)) {
        return 1;
    }

    // Initialize centroids by uniformly selecting K points from the dataset
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distrib(0, N - 1);

    std::cout << "Initializing centroids using random seed: " << seed << std::endl;
    for(int k = 0; k < K; ++k) {
        int random_idx = distrib(gen);
        for(int d = 0; d < D; ++d) {
            h_centroids[k * D + d] = h_points[d * N + random_idx]; 
        }
    }

    // Device memory allocation
    float *d_points, *d_centroids;
    int *d_assignments;
    CHECK_CUDA(cudaMalloc(&d_points, points_size));
    CHECK_CUDA(cudaMalloc(&d_centroids, centroids_size));
    CHECK_CUDA(cudaMalloc(&d_assignments, assignments_size));

    // Copy initial centroids to device
    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(), centroids_size, cudaMemcpyHostToDevice));
    
    // Calculate shared memory size needed for centroids
    size_t sharedMemSize = static_cast<size_t>(K) * D * sizeof(float);

    // Query Device Limits & Opt-In to Extended Shared Memory
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (sharedMemSize > prop.sharedMemPerBlockOptin) {
        std::cerr << "\n[ERROR] Required shared memory (" << sharedMemSize 
                  << " bytes) exceeds the absolute hardware limit for this GPU (" 
                  << prop.sharedMemPerBlockOptin << " bytes)." << std::endl;
        return 1;
    }

    CHECK_CUDA(cudaFuncSetAttribute(assignClustersKernel, 
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, 
                                    sharedMemSize));

    int minGridSize, blockSize = 0;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, 
        assignClustersKernel, 
        sharedMemSize, N)); 
        
    if (blockSize == 0) {
        std::cerr << "[ERROR] cudaOccupancyMaxPotentialBlockSize failed to assign a valid block size." << std::endl;
        return 1;
    }

    int gridSize = (N + blockSize - 1) / blockSize;

    int numBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, assignClustersKernel, blockSize, sharedMemSize);
    float occupancy = (numBlocksPerSM * blockSize) / (float)prop.maxThreadsPerMultiProcessor;

    std::cout << "--- Launch Configuration (Baseline) ---" << std::endl;
    std::cout << "N=" << N << " D=" << D << " K=" << K << std::endl;
    std::cout << "Block Size: " << blockSize << std::endl;
    std::cout << "Theoretical Occupancy: " << std::fixed << std::setprecision(2) << (occupancy * 100.0f) << "%" << std::endl;

    // Create CUDA events for profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event on the default stream
    cudaEventRecord(start, 0);

    // Copy points to device
    CHECK_CUDA(cudaMemcpy(d_points, h_points.data(), points_size, cudaMemcpyHostToDevice));

    // Launch the kernel
    assignClustersKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_points, d_centroids, d_assignments, N, K, D
    );

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_assignments.data(), d_assignments, assignments_size, cudaMemcpyDeviceToHost));

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

    std::cout << "\n--- Performance Metrics (Baseline) ---" << std::endl;
    std::cout << "Total Points (N): " << N << std::endl;
    std::cout << "Dimensions (D): " << D << std::endl;
    std::cout << "Clusters (K): " << K << std::endl;
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Algorithmic Effective Bandwidth: " << effective_bandwidth_GBs << " GB/s" << std::endl;
    std::cout << "Throughput: " << throughput_GFLOPS << " GFLOPS" << std::endl;

    // Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    CHECK_CUDA(cudaFree(d_points));
    CHECK_CUDA(cudaFree(d_centroids));
    CHECK_CUDA(cudaFree(d_assignments));

    return 0;
}