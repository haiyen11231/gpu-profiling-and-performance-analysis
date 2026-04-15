/**
 * Flash Attention — Basic HIP Implementation
 *
 * Direct port of the NVIDIA naive kernel to HIP. The algorithm is identical;
 * only the runtime API calls change (cuda* → hip*).
 *
 * Tiled forward pass with online softmax. Never materialises the full N×N
 * score matrix — memory is O(N) not O(N²).
 *
 * Assumptions: single head, single batch; N divisible by Br and Bc; FP32.
 *
 * Compile:
 *   hipcc -O3 --offload-arch=gfx90a naive.cpp -o naive_amd
 */

#include <hip/hip_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.h"

// Naive kernel uses its own (smaller) tile sizes: dim3(Bc, Br) must be ≤ 1024.
// Override the optimized-kernel Br/Bc with the naive-specific values.
#undef Br
#undef Bc
#define Br Br_NAIVE
#define Bc Bc_NAIVE

// Shared memory: Q(4KB) + K(8KB) + V(8KB) + S(2KB) ≈ 22 KB/block
// VGPRs:        O_acc[D=64] = 64 regs/thread × 512 threads/block ≈ 32K (fits)

#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t err = call;                                                \
        if (err != hipSuccess) {                                              \
            fprintf(stderr, "HIP error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, hipGetErrorString(err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


/**
 * flash_attention_forward
 *
 * Grid:  (N/Br) blocks — one per Q-tile
 * Block: (Bc, Br) — threadIdx.x = K column, threadIdx.y = Q row
 *
 * Each block loads its Q-tile once, then sweeps all K/V-tiles running
 * online softmax and accumulating the output in registers.
 */
__global__ __launch_bounds__(Br * Bc, 2)
void flash_attention_naive(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    float*       __restrict__ L,
    const int N
) {
    const int tile_row   = blockIdx.x;
    const int row        = threadIdx.y;
    const int col        = threadIdx.x;
    const int global_row = tile_row * Br + row;

    const float scale = 1.0f / sqrtf((float)D);

    __shared__ float Q_tile[Br][D];
    __shared__ float K_tile[Bc][D];
    __shared__ float V_tile[Bc][D];
    __shared__ float S_tile[Br][Bc];

    // Load Q-tile — stays resident for the full inner loop.
    for (int d_idx = col; d_idx < D; d_idx += Bc)
        Q_tile[row][d_idx] = Q[global_row * D + d_idx];
    __syncthreads();

    // Per-thread accumulators (in VGPRs).
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float O_acc[D];
    for (int d_idx = 0; d_idx < D; d_idx++)
        O_acc[d_idx] = 0.0f;

    const int num_kv_tiles = N / Bc;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {

        // Load K and V tiles.
        int kv_base = kv_tile * Bc;
        for (int d_idx = col; d_idx < D; d_idx += Bc) {
            K_tile[row][d_idx] = K[(kv_base + row) * D + d_idx];
            V_tile[row][d_idx] = V[(kv_base + row) * D + d_idx];
        }
        __syncthreads();

        // S[row][col] = dot(Q_tile[row], K_tile[col]) / sqrt(d)
        float score = 0.0f;
        for (int d_idx = 0; d_idx < D; d_idx++)
            score += Q_tile[row][d_idx] * K_tile[col][d_idx];
        S_tile[row][col] = score * scale;
        __syncthreads();

        // Online softmax: update running max and rescale.
        float row_max = -FLT_MAX;
        for (int j = 0; j < Bc; j++)
            row_max = fmaxf(row_max, S_tile[row][j]);

        float m_new   = fmaxf(m_i, row_max);
        float rescale = expf(m_i - m_new);

        float local_sum = 0.0f;
        for (int j = 0; j < Bc; j++)
            local_sum += expf(S_tile[row][j] - m_new);

        l_i = rescale * l_i + local_sum;

        // O += P · V; P recomputed from S_tile on the fly to avoid spilling
        // Bc extra registers to scratch memory (which maps to HBM).
        for (int d_idx = 0; d_idx < D; d_idx++) {
            O_acc[d_idx] *= rescale;
            for (int j = 0; j < Bc; j++)
                O_acc[d_idx] += expf(S_tile[row][j] - m_new) * V_tile[j][d_idx];
        }

        m_i = m_new;
        __syncthreads();
    }

    // Normalize and write O and logsumexp L back to HBM.
    float inv_l = 1.0f / l_i;
    for (int d_idx = 0; d_idx < D; d_idx++)
        O_acc[d_idx] *= inv_l;

    if (col == 0) {
        for (int d_idx = 0; d_idx < D; d_idx++)
            O[global_row * D + d_idx] = O_acc[d_idx];
        L[global_row] = m_i + logf(l_i);
    }
}


// CPU reference: O = softmax(Q·Kᵀ / sqrt(d)) · V — O(N²) memory, correctness checks only.
void naive_attention_cpu(
    const float* Q, const float* K, const float* V,
    float* O, int N, int d
) {
    float scale = 1.0f / sqrtf((float)d);
    float* S = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++)
                sum += Q[i * d + k] * K[j * d + k];
            S[i * N + j] = sum * scale;
        }

    for (int i = 0; i < N; i++) {
        float row_max = -FLT_MAX;
        for (int j = 0; j < N; j++)
            if (S[i * N + j] > row_max) row_max = S[i * N + j];
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            S[i * N + j] = expf(S[i * N + j] - row_max);
            row_sum += S[i * N + j];
        }
        for (int j = 0; j < N; j++)
            S[i * N + j] /= row_sum;
        for (int dd = 0; dd < d; dd++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++)
                sum += S[i * N + j] * V[j * d + dd];
            O[i * d + dd] = sum;
        }
    }
    free(S);
}


int main(int argc, char* argv[]) {
    int N = SEQ_LEN;
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0 || N % Br != 0 || N % Bc != 0) {
            fprintf(stderr, "Error: N must be a positive multiple of Br=%d and Bc=%d\n", Br, Bc);
            return EXIT_FAILURE;
        }
    }
    const int d = D;

    printf("Flash Attention — Naive HIP\n");
    printf("  N=%d  d=%d  Br=%d  Bc=%d\n\n", N, d, Br, Bc);

    size_t mat_size = N * d * sizeof(float);
    size_t vec_size = N * sizeof(float);

    float *h_Q = (float*)malloc(mat_size);
    float *h_K = (float*)malloc(mat_size);
    float *h_V = (float*)malloc(mat_size);
    float *h_O = (float*)malloc(mat_size);
#ifndef SKIP_CPU_VERIFY
    float *h_O_ref = (float*)malloc(mat_size);
#endif
    float *h_L = (float*)malloc(vec_size);

    srand(42);
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
    }

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    HIP_CHECK(hipMalloc(&d_Q, mat_size));
    HIP_CHECK(hipMalloc(&d_K, mat_size));
    HIP_CHECK(hipMalloc(&d_V, mat_size));
    HIP_CHECK(hipMalloc(&d_O, mat_size));
    HIP_CHECK(hipMalloc(&d_L, vec_size));

    HIP_CHECK(hipMemcpy(d_Q, h_Q, mat_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K, mat_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V, mat_size, hipMemcpyHostToDevice));

    dim3 grid(N / Br);
    dim3 block(Bc, Br);

    // Warmup then timed run.
    flash_attention_naive<<<grid, block>>>(d_Q, d_K, d_V, d_O, d_L, N);
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    flash_attention_naive<<<grid, block>>>(d_Q, d_K, d_V, d_O, d_L, N);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

    double total_flops = 4.0 * N * N * d;
    double tflops = (total_flops / (elapsed_ms * 1e-3)) / 1e12;

    printf("Kernel time:  %.4f ms\n", elapsed_ms);
    printf("FLOPs:        %.2e\n", total_flops);
    printf("Throughput:   %.4f TFLOP/s\n\n", tflops);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(h_O, d_O, mat_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_L, d_L, vec_size, hipMemcpyDeviceToHost));

#ifndef SKIP_CPU_VERIFY
    printf("Running CPU reference...\n");
    naive_attention_cpu(h_Q, h_K, h_V, h_O_ref, N, d);

    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    for (int i = 0; i < N * d; i++) {
        float diff = fabsf(h_O[i] - h_O_ref[i]);
        if (diff > max_diff) max_diff = diff;
        avg_diff += diff;
    }
    avg_diff /= (N * d);

    printf("CPU reference check:\n");
    printf("  Max error: %.6e\n", max_diff);
    printf("  Avg error: %.6e\n", avg_diff);
    printf("  %s\n", max_diff < 1e-3f ? "PASS" : "FAIL");
#endif

    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_L);
#ifndef SKIP_CPU_VERIFY
    free(h_O_ref);
#endif
    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_O); hipFree(d_L);

    return 0;
}
