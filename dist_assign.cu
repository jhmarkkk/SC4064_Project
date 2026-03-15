#include <cuda_runtime.h>
#include <iostream>
#include <cfloat>
#include <vector>
#include <random>

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
                float diff = points[point_idx * D + d] - s_centroids[k * D + d];
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


int main() {
    // Define N, K, and D for a quick test
    int N = 10000000; // Number of points
    int K = 10;       // Number of clusters
    int D = 8;        // Dimensions

    size_t points_size = N * D * sizeof(float);
    size_t centroids_size = K * D * sizeof(float);
    size_t assignments_size = N * sizeof(int);

    // Host memory allocation
    std::vector<float> h_points(N * D);
    std::vector<float> h_centroids(K * D);
    std::vector<int> h_assignments(N);

    // Initialize with dummy data
    // TODO: Replace with actual CSV loading logic
    for(int i = 0; i < N * D; ++i) h_points[i] = static_cast<float>(rand()) / RAND_MAX;
    for(int i = 0; i < K * D; ++i) h_centroids[i] = static_cast<float>(rand()) / RAND_MAX;

    // Device memory allocation
    float *d_points, *d_centroids;
    int *d_assignments;
    CHECK_CUDA(cudaMalloc(&d_points, points_size));
    CHECK_CUDA(cudaMalloc(&d_centroids, centroids_size));
    CHECK_CUDA(cudaMalloc(&d_assignments, assignments_size));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_points, h_points.data(), points_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(), centroids_size, cudaMemcpyHostToDevice));

    // Setup execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Calculate shared memory size needed for centroids
    size_t sharedMemSize = K * D * sizeof(float);

    // Calculate Occupancy
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int numBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, assignClustersKernel, threadsPerBlock, sharedMemSize);
    float occupancy = (numBlocksPerSM * threadsPerBlock) / (float)prop.maxThreadsPerMultiProcessor;

    std::cout << "--- Launch Configuration (Baseline) ---" << std::endl;
    std::cout << "Block Size: " << threadsPerBlock << std::endl;
    std::cout << "Theoretical Occupancy: " << std::fixed << std::setprecision(2) << (occupancy * 100.0f) << "%" << std::endl;

    // Create CUDA events for profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event on the default stream
    cudaEventRecord(start, 0);

    // Launch the kernel
    assignClustersKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
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