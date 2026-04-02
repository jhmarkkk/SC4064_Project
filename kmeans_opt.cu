#include "sbin.hh"
#include "save_results.hh"

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

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// =============================================================================
// Kernel 1: Optimised Cluster Assignment
// Chunked execution on global SoA + shared memory for centroids.
// Pinned memory + CUDA streams allow async overlap of transfer and compute.
// =============================================================================
__global__ void assignClustersKernel_Opt(
    const float* __restrict__ points_SoA,
    const float* __restrict__ centroids,
    int*   __restrict__ assignments,
    int chunk_size, int K, int D, int N, int offset)
{
    extern __shared__ float s_centroids[];

    // Collaboratively load all centroids into shared memory
    int total_centroid_elements = K * D;
    for (int i = threadIdx.x; i < total_centroid_elements; i += blockDim.x)
        s_centroids[i] = centroids[i];

    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_size) return;

    int global_idx = offset + idx;
    float min_dist    = FLT_MAX;
    int   best_cluster = -1;

    for (int k = 0; k < K; ++k) {
        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = points_SoA[d * N + global_idx] - s_centroids[k * D + d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist     = dist;
            best_cluster = k;
        }
    }

    assignments[global_idx] = best_cluster;
}

// =============================================================================
// Kernel 2: Optimised Centroid Recalculation — Shared Memory Reduction
//
// Why this is better than naive atomics:
//   Naive: every thread issues K*D atomicAdds directly to global memory.
//          With N threads all hitting the same K locations, this serialises.
//   Optimised: each block first accumulates into a shared memory buffer
//          (one per cluster per dimension), then ONE thread per block issues
//          the global atomicAdd. This reduces global atomic contention by
//          a factor of blockDim.x (typically 256-1024x fewer global atomics).
//
// Shared memory layout: s_sums[K * D] + s_counts[K]
//   s_sums  : partial sum of coordinates for each (cluster, dimension)
//   s_counts: partial count of points per cluster
// =============================================================================
__global__ void recalcCentroidsKernel_Opt(
    const float* __restrict__ points_SoA,
    const int*   __restrict__ assignments,
          float* __restrict__ centroid_sums,
          int*   __restrict__ centroid_counts,
    int N, int K, int D)
{
    // Dynamic shared memory layout:
    //   [0      .. K*D-1]  → partial sums   (float, K*D elements)
    //   [K*D    .. K*D+K-1]→ partial counts (int,   K elements)
    extern __shared__ float s_mem[];
    float* s_sums   = s_mem;
    int*   s_counts = reinterpret_cast<int*>(s_mem + K * D);

    // Zero the shared accumulators — every thread clears a portion
    for (int i = threadIdx.x; i < K * D; i += blockDim.x)
        s_sums[i] = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        s_counts[i] = 0;

    __syncthreads();

    // Each thread accumulates its point into shared memory (no contention —
    // only one thread per block writes to each shared location at a time
    // because each thread owns a unique point)
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx < N) {
        int cluster = assignments[point_idx];

        for (int d = 0; d < D; ++d)
            atomicAdd(&s_sums[cluster * D + d], points_SoA[d * N + point_idx]);

        atomicAdd(&s_counts[cluster], 1);
    }

    __syncthreads();

    // One flush per block: push shared partial results to global memory.
    // Only K*D + K global atomics per block instead of N*D per block.
    for (int i = threadIdx.x; i < K * D; i += blockDim.x)
        atomicAdd(&centroid_sums[i], s_sums[i]);

    for (int i = threadIdx.x; i < K; i += blockDim.x)
        atomicAdd(&centroid_counts[i], s_counts[i]);
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
        std::cerr << "Usage: " << argv[0]
                  << " <data_bin_file> [K_override] [Seed] [max_iters]\n";
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

    size_t total_points_bytes  = static_cast<size_t>(N) * D * sizeof(float);
    size_t total_assign_bytes  = static_cast<size_t>(N) * sizeof(int);
    size_t centroids_bytes     = static_cast<size_t>(K) * D * sizeof(float);

    // ── Pinned host allocations (enables async DMA transfers) ──
    float *h_points_SoA, *h_centroids;
    int   *h_assignments;
    CHECK_CUDA(cudaMallocHost(&h_points_SoA,  total_points_bytes));
    CHECK_CUDA(cudaMallocHost(&h_centroids,    centroids_bytes));
    CHECK_CUDA(cudaMallocHost(&h_assignments,  total_assign_bytes));

    if (!loadSbinSoA(argv[1], h_points_SoA, N, D)) return 1;

    // ── Initialise centroids: random subset of points ──
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distrib(0, N - 1);

    std::cout << "Initializing centroids using random seed: " << seed << std::endl;
    for (int k = 0; k < K; ++k) {
        int idx = distrib(gen);
        for (int d = 0; d < D; ++d)
            h_centroids[k * D + d] = h_points_SoA[d * N + idx];
    }

    // ── Device allocations ──
    float *d_points_SoA, *d_centroids, *d_centroid_sums;
    int   *d_assignments, *d_centroid_counts;

    CHECK_CUDA(cudaMalloc(&d_points_SoA,    total_points_bytes));
    CHECK_CUDA(cudaMalloc(&d_centroids,      centroids_bytes));
    CHECK_CUDA(cudaMalloc(&d_assignments,    total_assign_bytes));
    CHECK_CUDA(cudaMalloc(&d_centroid_sums,  centroids_bytes));
    CHECK_CUDA(cudaMalloc(&d_centroid_counts, K * sizeof(int)));

    // Copy full point dataset to device once — stays resident throughout
    CHECK_CUDA(cudaMemcpy(d_points_SoA, h_points_SoA,
                          total_points_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids,
                          centroids_bytes, cudaMemcpyHostToDevice));

    // ── Kernel configuration ──
    size_t assignSharedMem = static_cast<size_t>(K) * D * sizeof(float);
    // recalc shared mem: K*D floats for sums + K ints for counts
    size_t recalcSharedMem = static_cast<size_t>(K) * D * sizeof(float)
                           + static_cast<size_t>(K) * sizeof(int);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (assignSharedMem > prop.sharedMemPerBlockOptin) {
        std::cerr << "[ERROR] Shared memory required (" << assignSharedMem
                  << " bytes) exceeds hardware limit ("
                  << prop.sharedMemPerBlockOptin << " bytes)." << std::endl;
        return 1;
    }

    CHECK_CUDA(cudaFuncSetAttribute(assignClustersKernel_Opt,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    assignSharedMem));
    CHECK_CUDA(cudaFuncSetAttribute(recalcCentroidsKernel_Opt,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    recalcSharedMem));

    int minGridSize, blockSize = 0;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, assignClustersKernel_Opt, assignSharedMem, N));

    if (blockSize == 0) {
        std::cerr << "[ERROR] Could not determine block size." << std::endl;
        return 1;
    }

    int gridSize = (N + blockSize - 1) / blockSize;

    int numBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM, assignClustersKernel_Opt, blockSize, assignSharedMem);
    float occupancy = (numBlocksPerSM * blockSize) / (float)prop.maxThreadsPerMultiProcessor;

    std::cout << "--- Launch Configuration ---" << std::endl;
    std::cout << "N=" << N << "  D=" << D << "  K=" << K
              << "  max_iters=" << max_iters << std::endl;
    std::cout << "Block size            : " << blockSize << std::endl;
    std::cout << "Theoretical occupancy : " << std::fixed << std::setprecision(2)
              << (occupancy * 100.0f) << "%" << std::endl;
    std::cout << "Assign shared mem     : " << assignSharedMem << " bytes" << std::endl;
    std::cout << "Recalc shared mem     : " << recalcSharedMem << " bytes" << std::endl;

    // ── Stream sweep to find optimal stream count ──
    std::vector<int> test_streams = {1, 2, 4, 8, 16, 32};
    float best_time         = FLT_MAX;
    int   best_stream_count = -1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int num_streams : test_streams) {
        std::cout << "\nTesting NUM_STREAMS = " << num_streams << " ..." << std::endl;

        // Chunk layout
        std::vector<int>    stream_sizes(num_streams, 0);
        std::vector<size_t> stream_offsets(num_streams, 0);
        size_t cur_offset  = 0;
        int    base_chunk  = N / num_streams;
        int    remainder   = N % num_streams;
        for (int i = 0; i < num_streams; ++i) {
            stream_sizes[i]   = base_chunk + (i < remainder ? 1 : 0);
            stream_offsets[i] = cur_offset;
            cur_offset       += stream_sizes[i];
        }

        std::vector<cudaStream_t> streams(num_streams);
        for (int i = 0; i < num_streams; ++i)
            CHECK_CUDA(cudaStreamCreate(&streams[i]));

        cudaEventRecord(start, 0);

        // --- Copy points over once ---
        for (int i = 0; i < num_streams; ++i) {
            int    chunk  = stream_sizes[i];
            size_t offset = stream_offsets[i];
            if (chunk == 0) continue;
            size_t width_bytes = static_cast<size_t>(chunk) * sizeof(float);
            size_t pitch_bytes = static_cast<size_t>(N) * sizeof(float);

            // Async H2D transfer for this chunk (SoA strided copy)
            CHECK_CUDA(cudaMemcpy2DAsync(
                d_points_SoA + offset, pitch_bytes,
                h_points_SoA + offset, pitch_bytes,
                width_bytes, D,
                cudaMemcpyHostToDevice, streams[i]));

            }
        CHECK_CUDA(cudaDeviceSynchronize());  // wait for all chunks uploaded

        // ── K-means iteration loop ──
        for (int iter = 0; iter < max_iters; ++iter) {

            // --- Step 1: Chunked async assignment across streams ---
            for (int i = 0; i < num_streams; ++i) {
                int    chunk  = stream_sizes[i];
                size_t offset = stream_offsets[i];
                if (chunk == 0) continue;

                int    gSize       = (chunk + blockSize - 1) / blockSize;

                // Assignment kernel
                assignClustersKernel_Opt<<<gSize, blockSize, assignSharedMem, streams[i]>>>(
                    d_points_SoA, d_centroids, d_assignments,
                    chunk, K, D, N, static_cast<int>(offset));

                // Async D2H result transfer
                CHECK_CUDA(cudaMemcpyAsync(
                    h_assignments + offset,
                    d_assignments  + offset,
                    static_cast<size_t>(chunk) * sizeof(int),
                    cudaMemcpyDeviceToHost, streams[i]));
            }

            // Sync before recalculation — all assignments must be complete
            CHECK_CUDA(cudaDeviceSynchronize());

            // --- Step 2: Zero centroid accumulators ---
            CHECK_CUDA(cudaMemset(d_centroid_sums,   0, centroids_bytes));
            CHECK_CUDA(cudaMemset(d_centroid_counts, 0, K * sizeof(int)));

            // --- Step 3: Optimised shared-memory-reduction centroid recalc ---
            // Single kernel over all N points — streams don't help here since
            // all blocks must write to the same K centroid locations
            recalcCentroidsKernel_Opt<<<gridSize, blockSize, recalcSharedMem>>>(
                d_points_SoA, d_assignments,
                d_centroid_sums, d_centroid_counts,
                N, K, D);

            CHECK_CUDA(cudaDeviceSynchronize());

            // --- Step 4: Updating Centroid Kernel ---
            int kd_total  = K * D;
            int kd_blocks = (kd_total + blockSize - 1) / blockSize;
            updateCentroidsKernel<<<kd_blocks, blockSize>>>(
                d_centroid_sums, d_centroid_counts, d_centroids, K, D);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        double seconds = ms / 1000.0;

        // ── Performance Metrics ──
        //
        // Timing brackets: cudaEventRecord(start) → cudaEventRecord(stop)
        // This includes the one-time point upload + all iteration kernels.
        //
        // ── Bandwidth ──
        //
        // One-time H2D point upload (inside timing, happens once per stream sweep):
        //   N*D floats uploaded to device
        //   Total: N * D * 4 bytes
        //
        // Kernel 1 — assignment (per iteration):
        //   Reads : N*D floats (points from HBM) + K*D floats (centroids, shared mem load)
        //   Writes: N ints (assignments to global memory)
        //   D2H   : N ints (async copy of assignments back to host per stream)
        //   Total : N*D*4 + K*D*4 + N*4 + N*4
        //
        // Kernel 2 — recalc (per iteration):
        //   Reads : N*D floats (points) + N ints (assignments)
        //   Writes: K*D floats (centroid sums) + K ints (centroid counts)
        //   Total : N*D*4 + N*4 + K*D*4 + K*4
        //
        // Kernel 3 — update (per iteration):
        //   Reads : K*D floats (sums) + K ints (counts)
        //   Writes: K*D floats (new centroids)
        //   Total : K*D*4 + K*4 + K*D*4

        double bytes_upload = (double)N * D * 4;  // one-time point upload

        double bytes_assign = (double)N * D * 4   // read points
                            + (double)K * D * 4   // read centroids (shared mem load)
                            + (double)N * 4       // write assignments
                            + (double)N * 4;      // D2H assignments copy per stream

        double bytes_recalc = (double)N * D * 4   // read points
                            + (double)N * 4       // read assignments
                            + (double)K * D * 4   // write centroid sums
                            + (double)K * 4;      // write centroid counts

        double bytes_update = (double)K * D * 4   // read centroid sums
                            + (double)K * 4       // read centroid counts
                            + (double)K * D * 4;  // write new centroids

        double total_bytes      = bytes_upload
                                + (bytes_assign + bytes_recalc + bytes_update) * max_iters;
        double effective_bw_GBs = (total_bytes / 1e9) / seconds;

        // ── FLOP count ──
        //
        // Kernel 1 — assignment (per iteration):
        //   Per point, per cluster, per dimension: subtract + multiply + add = 3 FLOPs
        //   Total: N * K * D * 3
        //
        // Kernel 2 — recalc (per iteration):
        //   Per point per dimension: 1 atomicAdd to shared mem = 1 FLOP
        //   Per point: 1 atomicAdd to shared count = 1 FLOP
        //   Total: N * D + N = N * (D + 1)
        //
        // Kernel 3 — update (per iteration):
        //   Per (cluster, dimension): 1 divide = 1 FLOP
        //   Total: K * D
        //
        // Note: flops_assign dominates heavily.
        //   Example N=5M, K=50, D=32:
        //   flops_assign = 24 billion vs flops_recalc = 165 million vs flops_update = 1600

        double flops_assign      = 3.0 * (double)N * K * D;
        double flops_recalc      = (double)N * (D + 1);
        double flops_update      = (double)K * D;
        double total_flops       = (flops_assign + flops_recalc + flops_update) * max_iters;
        double throughput_GFLOPS = (total_flops / 1e9) / seconds;

        std::cout << "--- Performance Metrics (streams=" << num_streams << ") ---" << std::endl;
        std::cout << "Total time         : " << ms << " ms" << std::endl;
        std::cout << "Time per iteration : " << ms / max_iters << " ms" << std::endl;
        std::cout << "Effective BW       : " << effective_bw_GBs << " GB/s" << std::endl;
        std::cout << "Throughput         : " << throughput_GFLOPS << " GFLOPS" << std::endl;

        if (ms < best_time) {
            best_time         = ms;
            best_stream_count = num_streams;
        }

        for (int i = 0; i < num_streams; ++i)
            CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    // ── Copy final results from device to host ──
    CHECK_CUDA(cudaMemcpy(h_centroids, d_centroids,
                        centroids_bytes, cudaMemcpyDeviceToHost));
    // h_assignments is already up to date — last iteration's D2H async copy
    // has completed before cudaEventRecord(stop) + cudaEventSynchronize(stop)

    // ── Wrap raw pointers in vectors for save functions ──
    std::vector<int>   v_assignments(h_assignments, h_assignments + N);
    std::vector<float> v_centroids(h_centroids,    h_centroids   + K * D);

    // ── Save results ──
    std::string input_path = argv[1];
    std::string base = input_path.substr(input_path.find_last_of("/\\") + 1);
    base = base.substr(0, base.find_last_of('.'));
    std::string tag    = "optimised";
    std::string outdir = "optimised_results";
    system(("mkdir -p " + outdir).c_str());
    std::string prefix = outdir + "/" + base + "_" + tag;

    saveAssignmentsCSV(v_assignments,        prefix + "_assignments.csv");
    saveCentroidsCSV  (v_centroids,  K, D,   prefix + "_centroids.csv");
    saveAssignmentsBin(v_assignments,        prefix + "_assignments.bin");
    saveCentroidsBin  (v_centroids,  K, D,   prefix + "_centroids.bin");

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "N=" << N << "  D=" << D << "  K=" << K << std::endl;
    std::cout << "Block size     : " << blockSize << std::endl;
    std::cout << "Best streams   : " << best_stream_count
              << "  (" << best_time << " ms total, "
              << best_time / max_iters << " ms/iter)" << std::endl;

    // ── Cleanup ──
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK_CUDA(cudaFree(d_points_SoA));
    CHECK_CUDA(cudaFree(d_centroids));
    CHECK_CUDA(cudaFree(d_assignments));
    CHECK_CUDA(cudaFree(d_centroid_sums));
    CHECK_CUDA(cudaFree(d_centroid_counts));
    CHECK_CUDA(cudaFreeHost(h_points_SoA));
    CHECK_CUDA(cudaFreeHost(h_centroids));
    CHECK_CUDA(cudaFreeHost(h_assignments));

    return 0;
}
