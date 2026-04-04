#pragma once

#include <cuda_runtime.h>

// Shared memory size for this kernel must be K * D * sizeof(float)
// — same as assignSharedMem, so no extra allocation needed.
__global__ static void computeInertiaKernel(
    const float*  __restrict__ points,
    const float*  __restrict__ centroids,
    const int*    __restrict__ assignments,
          double* __restrict__ inertia,
    int N, int K, int D)
{
    extern __shared__ float s_centroids[];

    int total = K * D;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        s_centroids[i] = centroids[i];

    __syncthreads();

    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= N) return;

    int cluster = assignments[point_idx];
    double sq_dist = 0.0;

    for (int d = 0; d < D; ++d) {
        double diff = (double)points[d * N + point_idx]
                    - (double)s_centroids[cluster * D + d];
        sq_dist += diff * diff;
    }

    atomicAdd(inertia, sq_dist);
}