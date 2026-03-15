#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cfloat>
#include <vector>
#include <random>
#include <cmath>

#define TILE_DIM 32

// CUDA Error Macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// Optimized CUDA Kernel: Chunked SoA + Shared Memory
__global__ void assignClustersKernel_Opt(
    const float* __restrict__ points_chunk, 
    const float* __restrict__ centroids, 
    int* __restrict__ assignments_chunk, 
    int chunk_size, int K, int D) 
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
        float min_dist = FLT_MAX;
        int best_cluster = -1;

        // 3. Calculate distance to all K centroids
        for (int k = 0; k < K; ++k) {
            float current_dist = 0.0f;

            for (int d = 0; d < D; ++d) {
                // Read from the newly transposed SoA layout specifically built for this chunk size
                float pt_dim_val = points_chunk[d * chunk_size + idx];

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

int main() {
    // Problem scale - Arbitrary values to test robustness
    int N = 10000000;
    int K = 10;      
    int D = 8;       

    // TODO: Stream configuration
    const int NUM_STREAMS = 4;

    // Arrays to hold dynamic sizes and offsets for each stream
    std::vector<int> stream_sizes(NUM_STREAMS, 0);
    std::vector<size_t> stream_float_offsets(NUM_STREAMS, 0);
    std::vector<size_t> stream_int_offsets(NUM_STREAMS, 0);

    size_t current_float_offset = 0;
    size_t current_int_offset = 0;

    // Distribute N points across streams, accounting for remainder
    int base_chunk = N / NUM_STREAMS;
    int remainder = N % NUM_STREAMS;

    for (int i = 0; i < NUM_STREAMS; ++i) {
        // First 'remainder' streams get one extra point
        stream_sizes[i] = base_chunk + (i < remainder ? 1 : 0);

        // Record starting offsets for memory pointers
        stream_float_offsets[i] = current_float_offset;
        stream_int_offsets[i] = current_int_offset;

        // Update cumulative offsets
        current_float_offset += stream_sizes[i] * D;
        current_int_offset += stream_sizes[i];
    }

    size_t total_points_bytes = N * D * sizeof(float);
    size_t total_assign_bytes = N * sizeof(int);
    size_t centroids_bytes = K * D * sizeof(float);

    // 1. Allocate PINNED Host Memory for asynchronous transfers
    float *h_points_AoS, *h_centroids;
    int *h_assignments;
    CHECK_CUDA(cudaMallocHost(&h_points_AoS, total_points_bytes));
    CHECK_CUDA(cudaMallocHost(&h_centroids, centroids_bytes));
    CHECK_CUDA(cudaMallocHost(&h_assignments, total_assign_bytes));

    // 2. Initialize Data in standard "AoS" layout (Mimicking a CSV load)
    // TODO: Replace with actual CSV loading logic
    for (size_t i = 0; i < (size_t)N * D; ++i) {
        h_points_AoS[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for(int i = 0; i < K * D; ++i) {
        h_centroids[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 3. Allocate Device Memory
    float *d_points_AoS, *d_points_SoA, *d_centroids;
    int *d_assignments;
    CHECK_CUDA(cudaMalloc(&d_points_AoS, total_points_bytes));
    CHECK_CUDA(cudaMalloc(&d_points_SoA, total_points_bytes));
    CHECK_CUDA(cudaMalloc(&d_centroids, centroids_bytes));
    CHECK_CUDA(cudaMalloc(&d_assignments, total_assign_bytes));

    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids, centroids_bytes, cudaMemcpyHostToDevice));

    // 4. Create CUDA Streams & cuBLAS Handles
    cudaStream_t streams[NUM_STREAMS];
    cublasHandle_t cublas_handles[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        cublasCreate(&cublas_handles[i]);
        // Bind each cuBLAS handle to existing asynchronous streams
        cublasSetStream(cublas_handles[i], streams[i]); 
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 5. Dynamic Kernel Launch Configuration
    int minGridSize, blockSize;
    size_t sharedMemSize = K * D * sizeof(float);

    // We base the occupancy calculation on the largest chunk size
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, 
        assignClustersKernel_Opt, 
        sharedMemSize, stream_sizes[0])); 

    // Calculate Occupancy
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 0 is device ID
    int numBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, assignClustersKernel_Opt, blockSize, sharedMemSize);

    float occupancy = (numBlocksPerSM * blockSize) / (float)prop.maxThreadsPerMultiProcessor;

    std::cout << "--- Launch Configuration (Optimized, cuBLAS transpose) ---" << std::endl;
    std::cout << "Block Size: " << blockSize << std::endl;
    std::cout << "Theoretical Occupancy: " << std::fixed << std::setprecision(2) << (occupancy * 100.0f) << "%" << std::endl;

    // Create CUDA events for profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event on the default stream
    cudaEventRecord(start, 0);

    // 7. Asynchronous Execution Pipeline (Transfer -> Transpose -> Calculate -> Retrieve)
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int current_chunk_size = stream_sizes[i];

        if (current_chunk_size == 0)
            continue; 

        size_t float_offset = stream_float_offsets[i];
        size_t int_offset = stream_int_offsets[i];

        size_t current_chunk_bytes = current_chunk_size * D * sizeof(float);
        size_t current_assign_bytes = current_chunk_size * sizeof(int);

        // Grid parameters
        int gridSize = (current_chunk_size + blockSize - 1) / blockSize;

        // Step A: Async Copy AoS chunk from Host to Device
        CHECK_CUDA(cudaMemcpyAsync(
            d_points_AoS + float_offset, 
            h_points_AoS + float_offset, 
            current_chunk_bytes, cudaMemcpyHostToDevice, streams[i]));

        // Step B: Transpose this chunk from AoS to SoA using cuBLAS
        // Because C++ is Row-Major and cuBLAS is Column-Major, we swap D and current_chunk_size
        cublasSgeam(
            cublas_handles[i],
            CUBLAS_OP_T,    // Transpose matrix A
            CUBLAS_OP_N,    // Don't transpose matrix B (we don't use B anyway)
            current_chunk_size, // Rows of output matrix (in cuBLAS's column-major view)
            D,                  // Cols of output matrix (in cuBLAS's column-major view)
            &alpha,
            d_points_AoS + float_offset, // A matrix
            D,                  // Leading dimension of A (columns in C++ row-major)
            &beta,
            d_points_AoS + float_offset, // B matrix (ignored as beta=0)
            current_chunk_size, // Leading dimension of B
            d_points_SoA + float_offset, // Output matrix C
            current_chunk_size  // Leading dimension of C
        );

        // Step C: Execute Distance Kernel on the transposed SoA chunk
        assignClustersKernel_Opt<<<gridSize, blockSize, sharedMemSize, streams[i]>>>(
            d_points_SoA + float_offset, 
            d_centroids, 
            d_assignments + int_offset, 
            current_chunk_size, K, D);

        // Step D: Async Result Copy
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
    // Transpose (Read + Write): 2 * N * D * 4 bytes
    // Kernel Read (Points): N * D * 4 bytes
    // Kernel Write (Assignments): N * 4 bytes
    // Device to Host: N * 4 bytes
    // Total Bytes = N * (16D + 8) bytes
    double total_bytes = (double)N * (16.0 * D + 8.0);
    double effective_bandwidth_GBs = (total_bytes / 1e9) / seconds;

    // Calculate throughput in GFLOPS:
    // a subtraction, a multiplication, and an addition for every dimension
    double total_flops = 3.0 * (double)N * K * D;
    double throughput_GFLOPS = (total_flops / 1e9) / seconds;

    std::cout << "\n--- Performance Metrics (Optimized) ---" << std::endl;
    std::cout << "Total Points (N): " << N << std::endl;
    std::cout << "Dimensions (D): " << D << std::endl;
    std::cout << "Clusters (K): " << K << std::endl;
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << effective_bandwidth_GBs << " GB/s" << std::endl;
    std::cout << "Throughput: " << throughput_GFLOPS << " GFLOPS" << std::endl;

    // 8. Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cublasDestroy(cublas_handles[i]);
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaFree(d_points_AoS)); CHECK_CUDA(cudaFree(d_points_SoA)); 
    CHECK_CUDA(cudaFree(d_centroids)); CHECK_CUDA(cudaFree(d_assignments));
    CHECK_CUDA(cudaFreeHost(h_points_AoS)); CHECK_CUDA(cudaFreeHost(h_centroids)); CHECK_CUDA(cudaFreeHost(h_assignments));

    return 0;
}