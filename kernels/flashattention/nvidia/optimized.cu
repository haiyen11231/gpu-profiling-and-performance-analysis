#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.cuh"

using namespace nvcuda;


#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Shared memory size in bytes — used for both kernel attribute and launch.
// Layout (flat 1D, aligned to largest type = float):
//   Q [Br][D+PAD]  half
//   K [Bc][D+PAD]  half
//   V [Bc][D+PAD]  half
//   S [Br][Bc]     float
//   P [Br][Bc+PAD] half
//   O [Br][D]      float
//   m [Br]         float
//   l [Br]         float
static constexpr size_t SMEM_BYTES =
    sizeof(half)  * Br * (D + PAD) +
    sizeof(half)  * Bc * (D + PAD) +
    sizeof(half)  * Bc * (D + PAD) +
    sizeof(float) * Br * Bc        +
    sizeof(half)  * Br * (Bc + PAD)+
    sizeof(float) * Br * D         +
    sizeof(float) * Br * 2;


/**
 * flash_attention_optimized
 *
 * Grid:  (N/Br) blocks — one per Q-tile
 * Block: 128 threads (4 warps = NUM_WARPS), flat 1D
 *
 * FA2-style warp partitioning: warp i owns rows [i*WMMA_M : (i+1)*WMMA_M]
 * of S and O, so all four warps do useful WMMA work every iteration.
 */
__global__ __launch_bounds__(NUM_THREADS, 2)
void flash_attention_optimized(
    const half*  __restrict__ Q,
    const half*  __restrict__ K,
    const half*  __restrict__ V,
    float*       __restrict__ O,
    float*       __restrict__ L,
    const int N
) {
    const int tile_idx = blockIdx.x;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;

    const float scale = 1.0f / sqrtf((float)D);

    // Dynamic shared memory layout
    extern __shared__ char smem_raw[];
    half  *Q_smem = (half  *) smem_raw;
    half  *K_smem = Q_smem + Br * (D + PAD);
    half  *V_smem = K_smem + Bc * (D + PAD);
    float *S_smem = (float *)(V_smem + Bc * (D + PAD));
    half  *P_smem = (half  *)(S_smem + Br * Bc);
    float *O_smem = (float *)(P_smem + Br * (Bc + PAD));
    float *m_smem =           O_smem + Br * D;
    float *l_smem =           m_smem + Br;

    // Load Q-tile (stays resident for the full KV loop)
    const int q_row_base   = tile_idx * Br;
    const int q_tile_elems = Br * (D + PAD);
    for (int idx = tid; idx < q_tile_elems; idx += NUM_THREADS) {
        int r = idx / (D + PAD);
        int c = idx % (D + PAD);
        Q_smem[idx] = (c < D) ? Q[(q_row_base + r) * D + c] : __float2half(0.0f);
    }

    // ── Initialise O accumulator and softmax stats
    for (int idx = tid; idx < Br * D; idx += NUM_THREADS)
        O_smem[idx] = 0.0f;
    if (tid < Br) {
        m_smem[tid] = -FLT_MAX;
        l_smem[tid] = 0.0f;
    }
    __syncthreads();

    // ── Main loop: sweep all K/V-tiles
    const int num_kv_tiles = N / Bc;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {

        // Load K and V tiles into shared memory.
        const int kv_row_base   = kv_tile * Bc;
        const int kv_tile_elems = Bc * (D + PAD);
        for (int idx = tid; idx < kv_tile_elems; idx += NUM_THREADS) {
            int r = idx / (D + PAD);
            int c = idx % (D + PAD);
            if (c < D) {
                K_smem[idx] = K[(kv_row_base + r) * D + c];
                V_smem[idx] = V[(kv_row_base + r) * D + c];
            } else {
                K_smem[idx] = __float2half(0.0f);
                V_smem[idx] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // FA2: QK^T — all warps contribute
        // Warp i computes S[row_base : row_base+WMMA_M, 0 : Bc].
        // K_smem is [Bc][D+PAD] row-major; col_major load gives K^T for free:
        //   k_frag B[i][j] = K_smem[col+j][k+i] = K^T[k+i, col+j].
        {
            const int row_base = warp_id * WMMA_M;

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::col_major> k_frag;

            for (int col = 0; col < Bc; col += WMMA_N) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                               float> s_frag;
                wmma::fill_fragment(s_frag, 0.0f);

                for (int k = 0; k < D; k += WMMA_K) {
                    wmma::load_matrix_sync(q_frag,
                        Q_smem + row_base * (D + PAD) + k, D + PAD);
                    wmma::load_matrix_sync(k_frag,
                        K_smem + col * (D + PAD) + k,      D + PAD);
                    wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
                }
                wmma::store_matrix_sync(
                    S_smem + row_base * Bc + col, s_frag, Bc,
                    wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Online softmax (one thread per Q-row, tid < Br)
        if (tid < Br) {
            const int row = tid;
            float m_old = m_smem[row];
            float l_old = l_smem[row];

            // Scale S and find row max.
            float row_max = -FLT_MAX;
            for (int j = 0; j < Bc; j++) {
                S_smem[row * Bc + j] *= scale;
                row_max = fmaxf(row_max, S_smem[row * Bc + j]);
            }

            float m_new   = fmaxf(m_old, row_max);
            float rescale = expf(m_old - m_new);

            // Rescale accumulated O row for the updated max.
            for (int d_idx = 0; d_idx < D; d_idx++)
                O_smem[row * D + d_idx] = __fmaf_rn(rescale, O_smem[row * D + d_idx], 0.0f);

            // Compute P = exp(S - m_new); store as FP16 for WMMA.
            float local_sum = 0.0f;
            for (int j = 0; j < Bc; j++) {
                float p_val = expf(S_smem[row * Bc + j] - m_new);
                P_smem[row * (Bc + PAD) + j] = __float2half(p_val);
                local_sum += p_val;
            }
            for (int j = Bc; j < Bc + PAD; j++)
                P_smem[row * (Bc + PAD) + j] = __float2half(0.0f);

            // l_new = rescale * l_old + local_sum  (FMA: a*b+c, c ≠ 0)
            l_smem[row] = __fmaf_rn(rescale, l_old, local_sum);
            m_smem[row] = m_new;
        }
        __syncthreads();

        // FA2: O += P·V — all warps contribute
        // Warp i computes O[row_base : row_base+WMMA_M, 0 : D].
        // Accumulator seeded from O_smem (already rescaled above).
        {
            const int row_base = warp_id * WMMA_M;

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> v_frag;

            for (int col = 0; col < D; col += WMMA_N) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                               float> o_frag;
                wmma::load_matrix_sync(o_frag,
                    O_smem + row_base * D + col, D, wmma::mem_row_major);

                for (int k = 0; k < Bc; k += WMMA_K) {
                    wmma::load_matrix_sync(p_frag,
                        P_smem + row_base * (Bc + PAD) + k, Bc + PAD);
                    wmma::load_matrix_sync(v_frag,
                        V_smem + k * (D + PAD) + col,       D + PAD);
                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }
                wmma::store_matrix_sync(
                    O_smem + row_base * D + col, o_frag, D,
                    wmma::mem_row_major);
            }
        }
        __syncthreads();

    } // end KV-tile loop

    // Final normalisation and write back to HBM
    for (int idx = tid; idx < Br * D; idx += NUM_THREADS) {
        int r = idx / D;
        float inv_l = 1.0f / l_smem[r];
        O[(tile_idx * Br + r) * D + (idx % D)] = O_smem[idx] * inv_l;
    }
    if (tid < Br)
        L[tile_idx * Br + tid] = m_smem[tid] + logf(l_smem[tid]);
}


// CPU reference: O = softmax(Q·Kᵀ / sqrt(d)) · V — O(N²) memory, correctness checks only.
void naive_attention_cpu(
    const float* Q, const float* K, const float* V,
    float* O, int N, int d
) {
    float scale_val = 1.0f / sqrtf((float)d);
    float* S = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++)
                sum += Q[i * d + k] * K[j * d + k];
            S[i * N + j] = sum * scale_val;
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

    printf("Flash Attention — FA2-style FMA + WMMA\n");
    printf("  N=%d  d=%d  Br=%d  Bc=%d  WMMA=%dx%dx%d  threads=%d\n\n",
           N, d, Br, Bc, WMMA_M, WMMA_N, WMMA_K, NUM_THREADS);

    // Request extended shared memory for the larger tiles.
    CUDA_CHECK(cudaFuncSetAttribute(
        flash_attention_optimized,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_BYTES));

    size_t mat_size_f32 = (size_t)N * d * sizeof(float);
    size_t mat_size_f16 = (size_t)N * d * sizeof(half);
    size_t vec_size     = (size_t)N * sizeof(float);

    float* h_Q_f32 = (float*)malloc(mat_size_f32);
    float* h_K_f32 = (float*)malloc(mat_size_f32);
    float* h_V_f32 = (float*)malloc(mat_size_f32);
    float* h_O     = (float*)malloc(mat_size_f32);
#ifndef SKIP_CPU_VERIFY
    float* h_O_ref = (float*)malloc(mat_size_f32);
#endif
    float* h_L     = (float*)malloc(vec_size);
    half*  h_Q_f16 = (half*)malloc(mat_size_f16);
    half*  h_K_f16 = (half*)malloc(mat_size_f16);
    half*  h_V_f16 = (half*)malloc(mat_size_f16);

    srand(42);
    for (int i = 0; i < N * d; i++) {
        float qv = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        float kv = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        float vv = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        h_Q_f32[i] = qv;  h_Q_f16[i] = __float2half(qv);
        h_K_f32[i] = kv;  h_K_f16[i] = __float2half(kv);
        h_V_f32[i] = vv;  h_V_f16[i] = __float2half(vv);
    }

    half  *d_Q, *d_K, *d_V;
    float *d_O, *d_L;
    CUDA_CHECK(cudaMalloc(&d_Q, mat_size_f16));
    CUDA_CHECK(cudaMalloc(&d_K, mat_size_f16));
    CUDA_CHECK(cudaMalloc(&d_V, mat_size_f16));
    CUDA_CHECK(cudaMalloc(&d_O, mat_size_f32));
    CUDA_CHECK(cudaMalloc(&d_L, vec_size));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q_f16, mat_size_f16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K_f16, mat_size_f16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V_f16, mat_size_f16, cudaMemcpyHostToDevice));

    dim3 grid(N / Br);
    dim3 block(NUM_THREADS);

    // Warmup then timed run.
    flash_attention_optimized<<<grid, block, SMEM_BYTES>>>(d_Q, d_K, d_V, d_O, d_L, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    flash_attention_optimized<<<grid, block, SMEM_BYTES>>>(d_Q, d_K, d_V, d_O, d_L, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    double total_flops = 4.0 * N * N * d;
    double tflops = (total_flops / (elapsed_ms * 1e-3)) / 1e12;

    printf("Kernel time:  %.4f ms\n", elapsed_ms);
    printf("FLOPs:        %.2e\n", total_flops);
    printf("Throughput:   %.4f TFLOP/s\n\n", tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_O, d_O, mat_size_f32, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_L, d_L, vec_size,     cudaMemcpyDeviceToHost));

#ifndef SKIP_CPU_VERIFY
    printf("Running CPU reference...\n");
    naive_attention_cpu(h_Q_f32, h_K_f32, h_V_f32, h_O_ref, N, d);

    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    for (int i = 0; i < N * d; i++) {
        float diff = fabsf(h_O[i] - h_O_ref[i]);
        if (diff > max_diff) max_diff = diff;
        avg_diff += diff;
    }
    avg_diff /= (N * d);

    // FP16 input introduces ~1e-3 quantisation error vs FP32 reference — expected.
    printf("CPU reference check:\n");
    printf("  Max error: %.6e\n", max_diff);
    printf("  Avg error: %.6e\n", avg_diff);
    printf("  %s\n", max_diff < 5e-2f ? "PASS" : "FAIL");
#endif

    free(h_Q_f32); free(h_K_f32); free(h_V_f32);
    free(h_Q_f16); free(h_K_f16); free(h_V_f16);
    free(h_O); free(h_L);
#ifndef SKIP_CPU_VERIFY
    free(h_O_ref);
#endif
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);

    return 0;
}
