/**
 * =============================================================================
 * FLASH ATTENTION — FMA + WMMA (Tensor Core) CUDA Implementation
 * =============================================================================
 *
 * This version upgrades the basic scalar kernel with two key optimizations:
 *
 *   1. WMMA (Warp Matrix Multiply-Accumulate) — Tensor Cores
 *      Both matmuls (S = Q·K^T and O += P·V) are offloaded to Tensor Cores
 *      via the nvcuda::wmma API, replacing thousands of scalar FMAs with
 *      a handful of hardware-accelerated 16×16×16 matrix ops.
 *
 *   2. FMA (Fused Multiply-Add) intrinsics
 *      The online softmax rescaling uses __fmaf_rn() for single-rounding
 *      fused multiply-add where a*b+c naturally appears.
 *
 * Data types:
 *   - Inputs (Q, K, V):  FP16 (half) — required by Tensor Cores
 *   - Accumulation:       FP32 (float) — maintains numerical stability
 *   - This mixed-precision approach is standard in production transformers.
 *
 * Thread block design:
 *   - 128 threads = 4 warps per block
 *   - Warp 0:     computes S = Q·K^T via WMMA (one 16×16 output)
 *   - 16 threads:  scalar online softmax (one thread per Q-row)
 *   - Warps 0-3:  compute O += P·V via WMMA (each warp handles 16 of 64 cols)
 *   - All 128:    cooperative data loading from HBM → shared memory
 *
 * Requires: sm_70+ (Volta or newer) for FP16 Tensor Cores
 *
 * Compile:
 *   nvcc -O3 -arch=sm_70 flash_attention_wmma.cu -o flash_attention_wmma
 *
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>     // half, __float2half, __half2float
#include <mma.h>           // nvcuda::wmma API
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

// ---------------------------------------------------------------------------
// Tile and dimension constants
// ---------------------------------------------------------------------------
#define Br   16    // Q-tile rows (matches WMMA M dimension)
#define Bc   16    // K/V-tile rows (matches WMMA N dimension)
#define D    64   // Head dimension
#define PAD  8     // Shared memory padding (in elements) to avoid bank conflicts
                   //   Without padding: row of 64 halfs = 128 bytes = 32 banks
                   //   → every thread in a warp hits the same bank = 32-way conflict
                   //   With +8 halfs: row = 144 bytes, offsets shift per row

// WMMA tile shape: 16×16×16 (M × N × K)
// This is the native tile size for FP16 Tensor Cores on Volta/Turing/Ampere.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Thread block: 128 threads = 4 warps
#define NUM_THREADS 128
#define NUM_WARPS   (NUM_THREADS / 32)   // = 4

// ---------------------------------------------------------------------------
// Register budget check (128 threads/block):
//   WMMA fragments: ~8 regs each × 3 fragments = ~24 regs
//   Scalar locals:  ~15 regs
//   Total: ~39 regs/thread × 128 threads = ~5K registers/block
//   SM has 64K → excellent occupancy (up to 12+ blocks/SM)
//
// Shared memory usage:
//   Q_smem:  16 × 72 × 2 =  2,304 bytes
//   K_smem:  16 × 72 × 2 =  2,304 bytes
//   V_smem:  16 × 72 × 2 =  2,304 bytes
//   S_smem:  16 × 16 × 4 =  1,024 bytes
//   P_smem:  16 × 24 × 2 =    768 bytes
//   O_smem:  16 × 64 × 4 =  4,096 bytes
//   m/l:     16 × 4  × 2 =    128 bytes
//   Total:                  ~13 KB   (fits in 48 KB default shared memory)
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


/**
 * =============================================================================
 * KERNEL: flash_attention_fma_wmma_forward
 * =============================================================================
 *
 * Grid:   (N / Br)  — one block per Q-tile
 * Block:  (128)     — 4 warps, flat 1D indexing
 * =============================================================================
 */
__global__ __launch_bounds__(NUM_THREADS, 2)
void flash_attention_fma_wmma_forward(
    const half*  __restrict__ Q,   // [N, D] in FP16
    const half*  __restrict__ K,   // [N, D] in FP16
    const half*  __restrict__ V,   // [N, D] in FP16
    float*       __restrict__ O,   // [N, D] in FP32 (output)
    float*       __restrict__ L,   // [N]    logsumexp (for backward pass)
    const int N
) {
    // --- Thread identification ---
    const int tile_idx = blockIdx.x;               // which Q-tile (0 .. N/Br-1)
    const int tid      = threadIdx.x;              // flat thread id (0..127)
    const int warp_id  = tid / 32;                 // which warp (0..3)

    const float scale = 1.0f / sqrtf((float)D);

    // =====================================================================
    // Shared memory — all tiles for the block's working set
    // =====================================================================
    // PAD avoids shared memory bank conflicts for WMMA loads.
    // WMMA load_matrix_sync reads 16×16 sub-tiles; without padding,
    // threads in a warp map to the same bank, serializing accesses.
    __shared__ half  Q_smem[Br][D + PAD];       // Q tile (stays resident)
    __shared__ half  K_smem[Bc][D + PAD];       // K tile (swapped each iter)
    __shared__ half  V_smem[Bc][D + PAD];       // V tile (swapped each iter)
    __shared__ float S_smem[Br][Bc];            // Attention scores (FP32 for softmax)
    __shared__ half  P_smem[Br][Bc + PAD];      // Softmax output (FP16 for WMMA)
    __shared__ float O_smem[Br][D];             // Output accumulator (FP32)
    __shared__ float m_smem[Br];                // Running row-wise max
    __shared__ float l_smem[Br];                // Running row-wise sum

    // =====================================================================
    // STEP 1: Load Q-tile into shared memory (cooperative, all 128 threads)
    // =====================================================================
    const int q_row_base = tile_idx * Br;
    const int q_tile_elems = Br * (D + PAD);
    for (int idx = tid; idx < q_tile_elems; idx += NUM_THREADS) {
        int r = idx / (D + PAD);
        int c = idx % (D + PAD);
        // Load real data for c < D, zero-fill the padding columns
        Q_smem[r][c] = (c < D) ? Q[(q_row_base + r) * D + c]
                                : __float2half(0.0f);
    }

    // =====================================================================
    // STEP 2: Initialize accumulators
    // =====================================================================
    for (int idx = tid; idx < Br * D; idx += NUM_THREADS) {
        O_smem[idx / D][idx % D] = 0.0f;
    }
    if (tid < Br) {
        m_smem[tid] = -FLT_MAX;
        l_smem[tid] = 0.0f;
    }
    __syncthreads();

    // =====================================================================
    // STEP 3: Main loop — iterate over all K/V-tiles
    // =====================================================================
    const int num_kv_tiles = N / Bc;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {

        // -----------------------------------------------------------------
        // 3a. Load K-tile and V-tile (cooperative, all 128 threads)
        // -----------------------------------------------------------------
        const int kv_row_base = kv_tile * Bc;
        const int kv_tile_elems = Bc * (D + PAD);
        for (int idx = tid; idx < kv_tile_elems; idx += NUM_THREADS) {
            int r = idx / (D + PAD);
            int c = idx % (D + PAD);
            if (c < D) {
                K_smem[r][c] = K[(kv_row_base + r) * D + c];
                V_smem[r][c] = V[(kv_row_base + r) * D + c];
            } else {
                K_smem[r][c] = __float2half(0.0f);
                V_smem[r][c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // -----------------------------------------------------------------
        // 3b. WMMA: Compute S = Q · K^T  (Warp 0 only)
        // -----------------------------------------------------------------
        // S is Br×Bc = 16×16. The matmul inner dimension is D=64.
        // We iterate in chunks of WMMA_K=16: four mma_sync calls.
        //
        // Key insight for K^T:
        //   K_smem is stored row-major as [Bc][D+PAD], i.e., K[j][k].
        //   We load it as matrix_b with col_major layout.
        //   col_major load reads element [k'][j] = *(ptr + j*ldm + k')
        //                                        = K_smem[j][k_start + k']
        //   So fragment B[k'][j] = K[j][k_start+k'] = K^T[k_start+k'][j]
        //   → We get K^T without ever transposing in memory!
        if (warp_id == 0) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::col_major> k_frag;  // col_major → K^T
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float> s_frag;

            wmma::fill_fragment(s_frag, 0.0f);

            // Accumulate over D in chunks of 16
            for (int k = 0; k < D; k += WMMA_K) {
                // Q_slice: rows 0..15, cols k..k+15 of Q_smem
                wmma::load_matrix_sync(q_frag, &Q_smem[0][k], D + PAD);
                // K_slice: interpreted as K^T thanks to col_major
                wmma::load_matrix_sync(k_frag, &K_smem[0][k], D + PAD);

                // S += Q_slice × K_slice^T  (one Tensor Core instruction)
                // Replaces 16 × 16 × 16 = 4096 scalar FMA operations!
                wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
            }

            // Store 16×16 score matrix to shared memory (FP32)
            wmma::store_matrix_sync(&S_smem[0][0], s_frag, Bc,
                                    wmma::mem_row_major);
        }
        __syncthreads();

        // -----------------------------------------------------------------
        // 3c. Online softmax + rescale O  (16 threads, one per Q-row)
        // -----------------------------------------------------------------
        // Only threads 0..15 are active. Each handles one row of S_smem.
        // The other 112 threads wait at the syncthreads below.
        //
        // This is the scalar part of Flash Attention — can't use Tensor
        // Cores here because exp() and max() are element-wise, not matmuls.
        // FMA intrinsics are used where a*b+c naturally appears.
        if (tid < Br) {
            const int row = tid;
            float m_old = m_smem[row];
            float l_old = l_smem[row];

            // --- Apply attention scale: S = S / sqrt(d) ---
            // --- Simultaneously find row maximum ---
            float row_max = -FLT_MAX;
            for (int j = 0; j < Bc; j++) {
                S_smem[row][j] *= scale;
                row_max = fmaxf(row_max, S_smem[row][j]);
            }

            // --- Update running max ---
            float m_new = fmaxf(m_old, row_max);

            // --- Rescale factor: corrects all previous exp() values ---
            // When the max changes from m_old to m_new, every previous
            // exp(x - m_old) must be multiplied by exp(m_old - m_new)
            // to become exp(x - m_new).
            float rescale = expf(m_old - m_new);

            // --- Rescale the running O accumulator ---
            // O_new[d] = rescale * O_old[d]
            // We use __fmaf_rn(a, b, 0.0f) here. For a pure multiply,
            // this is identical to a*b in terms of result, but goes through
            // the FMA pipeline which may have better throughput on some
            // architectures when interleaved with other FMA ops.
            for (int d_idx = 0; d_idx < D; d_idx++) {
                O_smem[row][d_idx] = __fmaf_rn(rescale, O_smem[row][d_idx],
                                                0.0f);
            }

            // --- Compute P = exp(S - m_new) and local row sum ---
            // Also convert P to FP16 and store in P_smem for the
            // subsequent WMMA matmul (Tensor Cores require FP16 inputs).
            float local_sum = 0.0f;
            for (int j = 0; j < Bc; j++) {
                float p_val = expf(S_smem[row][j] - m_new);
                P_smem[row][j] = __float2half(p_val);
                local_sum += p_val;
            }
            // Zero-fill padding columns of P_smem
            for (int j = Bc; j < Bc + PAD; j++) {
                P_smem[row][j] = __float2half(0.0f);
            }

            // --- Update running sum using FMA ---
            // l_new = rescale * l_old + local_sum
            //       = exp(m_old - m_new) * l_old + sum_j(exp(S_j - m_new))
            //
            // THIS is the textbook FMA use case: a true a*b+c where
            // c ≠ 0 and single-rounding gives better numerical accuracy
            // than a separate multiply then add.
            l_smem[row] = __fmaf_rn(rescale, l_old, local_sum);
            m_smem[row] = m_new;
        }
        __syncthreads();

        // -----------------------------------------------------------------
        // 3d. WMMA: Compute O += P · V  (all 4 warps in parallel)
        // -----------------------------------------------------------------
        // O is Br×D = 16×64. We split D into 4 chunks of 16 columns.
        // Each warp handles one chunk: warp w → columns [w*16 : (w+1)*16].
        //
        // For each warp:
        //   A = P     (16×16, matrix_a, row_major) — same for all warps
        //   B = V_chunk (16×16, matrix_b, row_major) — different per warp
        //   C = O_chunk (16×16, accumulator) — load existing, accumulate, store
        //
        // The mma_sync call computes: C = A × B + C
        // Since C is loaded with the already-rescaled O values, this
        // naturally gives us: O_chunk = P × V_chunk + O_chunk_rescaled
        {
            const int col_offset = warp_id * 16;  // 0, 16, 32, or 48

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float> o_frag;

            // Load the current (rescaled) O chunk as the accumulator's
            // initial value — mma_sync will ADD P×V to this.
            wmma::load_matrix_sync(o_frag, &O_smem[0][col_offset], D,
                                   wmma::mem_row_major);

            // Load P: 16×16 softmax probabilities (same for all warps)
            wmma::load_matrix_sync(p_frag, &P_smem[0][0], Bc + PAD);

            // Load V chunk: 16 rows × 16 cols at the warp's column offset
            wmma::load_matrix_sync(v_frag, &V_smem[0][col_offset], D + PAD);

            // === THE TENSOR CORE OPERATION ===
            // O_chunk = P × V_chunk + O_chunk
            // One instruction replaces 16×16×16 = 4096 scalar FMAs.
            // On Volta: 64 TOPS for FP16; on Ampere: 312 TOPS.
            wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);

            // Store updated O chunk back to shared memory
            wmma::store_matrix_sync(&O_smem[0][col_offset], o_frag, D,
                                    wmma::mem_row_major);
        }
        __syncthreads();

    } // end KV-tile loop

    // =====================================================================
    // STEP 4: Final normalization and write back to HBM
    // =====================================================================
    // O_final = O_smem / l   (divide by softmax denominator)
    // L = m + log(l)         (logsumexp, needed for backward pass)
    for (int idx = tid; idx < Br * D; idx += NUM_THREADS) {
        int r = idx / D;
        int c = idx % D;
        float inv_l = 1.0f / l_smem[r];
        O[(tile_idx * Br + r) * D + c] = O_smem[r][c] * inv_l;
    }
    if (tid < Br) {
        L[tile_idx * Br + tid] = m_smem[tid] + logf(l_smem[tid]);
    }
}


// ===========================================================================
// REFERENCE: Naive attention on CPU (FP32) for correctness verification
// ===========================================================================
void naive_attention_cpu(
    const float* Q, const float* K, const float* V,
    float* O, int N, int d
) {
    float scale_val = 1.0f / sqrtf((float)d);
    float* S = (float*)malloc(N * N * sizeof(float));

    // S = (Q · K^T) / sqrt(d)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = sum * scale_val;
        }
    }

    // Softmax per row + output
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


// ===========================================================================
// MAIN
// ===========================================================================
int main() {
    const int N = (1 << 20);
    const int d = D;

    printf("Flash Attention — FMA + WMMA (Tensor Core) Implementation\n");
    printf("===========================================================\n");
    printf("  Sequence length  N  = %d\n", N);
    printf("  Head dimension   d  = %d\n", d);
    printf("  Q-tile rows     Br  = %d\n", Br);
    printf("  K-tile rows     Bc  = %d\n", Bc);
    printf("  WMMA tile shape      = %d × %d × %d\n", WMMA_M, WMMA_N, WMMA_K);
    printf("  Threads/block        = %d (%d warps)\n", NUM_THREADS, NUM_WARPS);
    printf("  Grid size            = %d blocks\n", N / Br);
    printf("  Precision            = FP16 input, FP32 accumulation\n\n");

    size_t mat_size_f32  = N * d * sizeof(float);
    size_t mat_size_f16  = N * d * sizeof(half);
    size_t vec_size      = N * sizeof(float);

    // --- Host allocation ---
    float* h_Q_f32     = (float*)malloc(mat_size_f32);
    float* h_K_f32     = (float*)malloc(mat_size_f32);
    float* h_V_f32     = (float*)malloc(mat_size_f32);
    float* h_O         = (float*)malloc(mat_size_f32);
    float* h_O_ref     = (float*)malloc(mat_size_f32);
    float* h_L         = (float*)malloc(vec_size);
    half*  h_Q_f16     = (half*)malloc(mat_size_f16);
    half*  h_K_f16     = (half*)malloc(mat_size_f16);
    half*  h_V_f16     = (half*)malloc(mat_size_f16);

    // --- Initialize with random values ---
    srand(42);
    for (int i = 0; i < N * d; i++) {
        float qv = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        float kv = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        float vv = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        h_Q_f32[i] = qv;  h_Q_f16[i] = __float2half(qv);
        h_K_f32[i] = kv;  h_K_f16[i] = __float2half(kv);
        h_V_f32[i] = vv;  h_V_f16[i] = __float2half(vv);
    }

    // --- Device allocation ---
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

    // --- Kernel launch ---
    dim3 grid(N / Br);
    dim3 block(NUM_THREADS);

    // =====================================================================
    // CUDA Event Timing
    // =====================================================================
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup (absorbs JIT, context init, page faults)
    flash_attention_fma_wmma_forward<<<grid, block>>>(d_Q, d_K, d_V,
                                                       d_O, d_L, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    flash_attention_fma_wmma_forward<<<grid, block>>>(d_Q, d_K, d_V,
                                                       d_O, d_L, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    double total_flops = 4.0 * N * N * d;
    double tflops = (total_flops / (elapsed_ms * 1e-3)) / 1e12;

    printf("--- Kernel Timing (CUDA Events) ---\n");
    printf("  Elapsed time:  %.4f ms\n", elapsed_ms);
    printf("  Approx FLOPs:  %.2e\n", total_flops);
    printf("  Throughput:    %.4f TFLOP/s\n\n", tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGetLastError());

    // --- Copy output back ---
    CUDA_CHECK(cudaMemcpy(h_O, d_O, mat_size_f32, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_L, d_L, vec_size,     cudaMemcpyDeviceToHost));

    // --- CPU reference (FP32 throughout) ---
    printf("Computing naive FP32 attention on CPU for verification...\n");
    naive_attention_cpu(h_Q_f32, h_K_f32, h_V_f32, h_O_ref, N, d);

    // --- Verify ---
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    for (int i = 0; i < N * d; i++) {
        float diff = fabsf(h_O[i] - h_O_ref[i]);
        if (diff > max_diff) max_diff = diff;
        avg_diff += diff;
    }
    avg_diff /= (N * d);

    printf("\nVerification against naive FP32 attention:\n");
    printf("  Max absolute error: %.6e\n", max_diff);
    printf("  Avg absolute error: %.6e\n", avg_diff);

    // FP16 inputs introduce quantization error. The reference uses FP32
    // throughout, so we expect ~1e-3 max error from half-precision rounding.
    // This is normal and acceptable for transformer inference.
    if (max_diff < 5e-2f) {
        printf("  PASSED — within expected FP16 tolerance\n");
    } else {
        printf("  FAILED — errors too large, check implementation\n");
    }

    // --- Cleanup ---
    free(h_Q_f32); free(h_K_f32); free(h_V_f32);
    free(h_Q_f16); free(h_K_f16); free(h_V_f16);
    free(h_O); free(h_O_ref); free(h_L);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);

    return 0;
}
