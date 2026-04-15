/**
 * =============================================================================
 * FLASH ATTENTION — Basic CUDA Implementation (No FMA / No WMMA)
 * =============================================================================
 *
 * Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
 *            Attention with IO-Awareness" (NeurIPS 2022)
 *
 * This is a teaching implementation. It is correct and demonstrates the core
 * algorithm, but omits production optimizations (tensor cores, async copies,
 * warp specialization, etc.)
 *
 * Memory complexity:  O(N)  — no N×N matrix ever materialized
 * IO complexity:      O(N² d / M)  where M = SRAM size
 *
 * Assumptions for simplicity:
 *   - Single-head, single-batch (extend by adding grid dims)
 *   - seq_len N is divisible by BLOCK_SIZE
 *   - head_dim d is divisible by BLOCK_SIZE
 *   - FP32 throughout (no mixed precision)
 *
 * Compile:
 *   nvcc -O3 -arch=sm_70 flash_attention.cu -o flash_attention
 *
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Tile sizes — chosen to fit in shared memory
//   Br = tile rows for Q  (each thread block processes Br query rows)
//   Bc = tile cols for K/V (inner loop iterates over Bc key rows at a time)
//   d  = head dimension (compile-time constant for simplicity)
// ---------------------------------------------------------------------------
#define Br 16    // Q block rows  (reduced from 32 to lower register pressure)
#define Bc 16    // K/V block rows (reduced from 32 to lower register pressure)
#define D  64    // head dimension (fixed for this example)

// Register budget check:
//   O_acc[D=64] = 64 regs/thread  (the main consumer)
//   + ~10 scalar regs
//   ≈ 74 regs/thread × 256 threads/block = ~19K registers
//   SM has 64K registers → comfortably fits, good occupancy
//
// Shared memory usage per block:
//   Q_tile:  Br × D  = 16×64 = 4 KB
//   K_tile:  Bc × D  = 16×64 = 4 KB
//   V_tile:  Bc × D  = 16×64 = 4 KB
//   S_tile:  Br × Bc = 16×16 = 1 KB
//   Total: ~13 KB — fits easily in 48 KB shared memory.

// ---------------------------------------------------------------------------
// Helper: check CUDA errors
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
 * KERNEL: flash_attention_forward
 * =============================================================================
 *
 * Grid:   (N / Br, 1, 1)  — one thread block per Q-tile
 * Block:  (Bc, Br, 1)     — threadIdx.x indexes the K/S column,
 *                            threadIdx.y indexes the Q/O row within the tile
 *
 * Each thread block:
 *   1. Loads its Q-tile into shared memory (constant for the block)
 *   2. Iterates over ALL K/V-tiles:
 *      a. Load K-tile, V-tile into shared memory
 *      b. Compute S = Q_tile · K_tile^T  (Br × Bc scores)
 *      c. Apply scaling by 1/sqrt(d)
 *      d. Online softmax: update running max (m) and sum (l)
 *      e. Accumulate output: O += P · V_tile
 *   3. Final normalization: O /= l
 *   4. Write O and logsumexp L back to HBM
 *
 * Parameters:
 *   Q, K, V  — [N, D] input matrices in HBM
 *   O        — [N, D] output matrix in HBM
 *   L        — [N]    logsumexp per row (needed for backward pass)
 *   N        — sequence length
 * =============================================================================
 */
__global__ __launch_bounds__(Br * Bc, 2)  // max 256 threads, aim for 2 blocks/SM
void flash_attention_forward(
    const float* __restrict__ Q,   // [N, D]
    const float* __restrict__ K,   // [N, D]
    const float* __restrict__ V,   // [N, D]
    float*       __restrict__ O,   // [N, D]
    float*       __restrict__ L,   // [N]
    const int N
) {
    // ----- Step 0: Identify which Q-tile this block owns -----
    const int tile_row = blockIdx.x;            // which Q-tile (0 .. N/Br - 1)
    const int row      = threadIdx.y;           // local row within Q-tile [0, Br)
    const int col      = threadIdx.x;           // used for K/S column    [0, Bc)
    const int global_row = tile_row * Br + row; // absolute row in Q/O

    // Scaling factor:  1 / sqrt(d)
    const float scale = 1.0f / sqrtf((float)D);

    // ----- Shared memory allocation -----
    // Q_tile stays resident for the entire inner loop.
    // K_tile and V_tile are overwritten each inner iteration.
    __shared__ float Q_tile[Br][D];
    __shared__ float K_tile[Bc][D];
    __shared__ float V_tile[Bc][D];
    __shared__ float S_tile[Br][Bc];  // attention scores for current K-block

    // =====================================================================
    // STEP 1: Load Q-tile into shared memory
    // =====================================================================
    // Each thread loads multiple elements along the D dimension.
    // threadIdx.y selects the row, we loop over d in strides of Bc.
    for (int d_idx = col; d_idx < D; d_idx += Bc) {
        Q_tile[row][d_idx] = Q[global_row * D + d_idx];
    }
    __syncthreads();

    // =====================================================================
    // STEP 2: Initialize online softmax accumulators (per row)
    // =====================================================================
    // m_i = running row-wise maximum of attention scores  (init: -inf)
    // l_i = running row-wise sum of exp(score - m)        (init: 0)
    // O_acc[d] = running unnormalized output accumulator  (init: 0)
    //
    // These live in REGISTERS — one set per thread (per Q-row).
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float O_acc[D];
    for (int d_idx = 0; d_idx < D; d_idx++) {
        O_acc[d_idx] = 0.0f;
    }

    // Total number of K/V-tiles to iterate over
    const int num_kv_tiles = N / Bc;

    // =====================================================================
    // STEP 3: Main loop — iterate over all K/V-tiles
    // =====================================================================
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {

        // --- 3a. Load K-tile and V-tile into shared memory ---
        // K_tile[col_k][d] = K[kv_tile * Bc + col_k, d]
        int kv_base = kv_tile * Bc;
        for (int d_idx = col; d_idx < D; d_idx += Bc) {
            K_tile[row][d_idx] = K[(kv_base + row) * D + d_idx];
            V_tile[row][d_idx] = V[(kv_base + row) * D + d_idx];
        }
        // NOTE: We reuse threadIdx.y (row) to load K/V rows.
        // When Br == Bc, each thread loads one row — perfect.
        // When Br != Bc, you'd need extra logic here.
        __syncthreads();

        // --- 3b. Compute S = Q_tile · K_tile^T  (score matrix) ---
        // Each thread computes S_tile[row][col] = dot(Q_tile[row], K_tile[col])
        // This is a Br × Bc matmul where the inner dimension is D.
        float score = 0.0f;
        for (int d_idx = 0; d_idx < D; d_idx++) {
            score += Q_tile[row][d_idx] * K_tile[col][d_idx];
        }

        // --- 3c. Apply scaling ---
        score *= scale;
        S_tile[row][col] = score;
        __syncthreads();

        // --- 3d. Online softmax (the heart of Flash Attention) ---
        //
        // We need to compute softmax across the FULL row of S (all K-tiles),
        // but we only see one K-tile at a time. The online algorithm maintains
        // running statistics m_i (max) and l_i (sum) that converge to the
        // correct values after all tiles are processed.
        //
        // Algorithm per row:
        //   m_new = max(m_old, max_j(S[row][j]))
        //   P[row][j] = exp(S[row][j] - m_new)          for this block
        //   l_new = exp(m_old - m_new) * l_old + sum_j(P[row][j])
        //   O = exp(m_old - m_new) * O + P · V_block
        //   m_old = m_new, l_old = l_new

        // Find row max of current S-tile block (reduction across col dimension)
        // Each thread has one column; we reduce across threadIdx.x.
        float row_max = -FLT_MAX;
        for (int j = 0; j < Bc; j++) {
            if (S_tile[row][j] > row_max) {
                row_max = S_tile[row][j];
            }
        }

        // Update running max
        float m_new = fmaxf(m_i, row_max);

        // Compute local row sum of exp(S - m_new)
        // NOTE: We do NOT store a P_row[Bc] array — that would burn Bc
        // registers per thread. Instead we recompute exp() from S_tile
        // (in shared memory, cheap to re-read) during the V accumulation.
        float local_sum = 0.0f;
        for (int j = 0; j < Bc; j++) {
            local_sum += expf(S_tile[row][j] - m_new);
        }

        // Rescale factor for previous accumulations
        //   When m changes, all previous exp() values were computed with the
        //   old max. Multiplying by exp(m_old - m_new) corrects them.
        float rescale = expf(m_i - m_new);

        // Update running sum:  l_new = rescale * l_old + local_sum
        l_i = rescale * l_i + local_sum;

        // --- 3e. Accumulate output ---
        // O_acc = rescale * O_acc + P · V_tile
        // Recompute P[j] = exp(S[row][j] - m_new) on the fly from S_tile.
        // Two reads of S_tile from SRAM (~20 TB/s) is far cheaper than
        // spilling Bc registers to local memory (which is actually HBM).
        for (int d_idx = 0; d_idx < D; d_idx++) {
            O_acc[d_idx] *= rescale;  // correct previous accumulation
            for (int j = 0; j < Bc; j++) {
                O_acc[d_idx] += expf(S_tile[row][j] - m_new) * V_tile[j][d_idx];
            }
        }

        // Update running max for next iteration
        m_i = m_new;

        __syncthreads();  // protect smem before next K/V tile load
    }

    // =====================================================================
    // STEP 4: Final normalization
    // =====================================================================
    // O_final = O_acc / l_i   (divide by the softmax denominator)
    float inv_l = 1.0f / l_i;
    for (int d_idx = 0; d_idx < D; d_idx++) {
        O_acc[d_idx] *= inv_l;
    }

    // =====================================================================
    // STEP 5: Write results back to HBM
    // =====================================================================
    // Store O row and logsumexp L (used in backward pass)
    //
    // Only one thread per row needs to do this. We use threadIdx.x == 0
    // since all threads in the same row computed the same O_acc.
    //
    // Actually: each thread (row, col) computed the SAME O_acc for its row
    // because the inner loops over j iterate over ALL Bc columns.
    // So any thread with the correct `row` can write. We pick col == 0.
    if (col == 0) {
        for (int d_idx = 0; d_idx < D; d_idx++) {
            O[global_row * D + d_idx] = O_acc[d_idx];
        }
        // L = m + log(l)  — the log-sum-exp, needed for backward pass
        L[global_row] = m_i + logf(l_i);
    }
}


// ===========================================================================
// REFERENCE: Naive attention for correctness verification
// ===========================================================================
// Computes O = softmax(Q·K^T / sqrt(d)) · V  on the CPU.
// This IS the O(N²) memory version — only used for testing.
void naive_attention_cpu(
    const float* Q, const float* K, const float* V,
    float* O, int N, int d
) {
    float scale = 1.0f / sqrtf((float)d);

    // Allocate full N×N score matrix (what Flash Attention avoids!)
    float* S = (float*)malloc(N * N * sizeof(float));

    // S = Q · K^T
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = sum * scale;
        }
    }

    // Softmax per row + output accumulation
    for (int i = 0; i < N; i++) {
        // Find max for numerical stability
        float row_max = -FLT_MAX;
        for (int j = 0; j < N; j++) {
            if (S[i * N + j] > row_max) row_max = S[i * N + j];
        }
        // Exp and sum
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            S[i * N + j] = expf(S[i * N + j] - row_max);
            row_sum += S[i * N + j];
        }
        // Normalize and compute O
        for (int j = 0; j < N; j++) {
            S[i * N + j] /= row_sum;
        }
        for (int dd = 0; dd < d; dd++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += S[i * N + j] * V[j * d + dd];
            }
            O[i * d + dd] = sum;
        }
    }

    free(S);
}


// ===========================================================================
// MAIN: Allocate, run Flash Attention kernel, verify against naive
// ===========================================================================
int main() {
    // --- Configuration ---
    const int N = 256;   // sequence length (must be divisible by Br and Bc)
    const int d = D;     // head dimension (must match compile-time D)

    printf("Flash Attention Demo\n");
    printf("  Sequence length N = %d\n", N);
    printf("  Head dimension  d = %d\n", d);
    printf("  Q-tile rows   Br = %d\n", Br);
    printf("  K-tile rows   Bc = %d\n", Bc);
    printf("  Number of Q-tiles = %d\n", N / Br);
    printf("  Number of K-tiles = %d\n\n", N / Bc);

    size_t mat_size = N * d * sizeof(float);
    size_t vec_size = N * sizeof(float);

    // --- Host allocation ---
    float *h_Q = (float*)malloc(mat_size);
    float *h_K = (float*)malloc(mat_size);
    float *h_V = (float*)malloc(mat_size);
    float *h_O = (float*)malloc(mat_size);       // Flash Attention output
    float *h_O_ref = (float*)malloc(mat_size);   // naive reference output
    float *h_L = (float*)malloc(vec_size);

    // --- Initialize with random values ---
    srand(42);
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
    }

    // --- Device allocation ---
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    CUDA_CHECK(cudaMalloc(&d_Q, mat_size));
    CUDA_CHECK(cudaMalloc(&d_K, mat_size));
    CUDA_CHECK(cudaMalloc(&d_V, mat_size));
    CUDA_CHECK(cudaMalloc(&d_O, mat_size));
    CUDA_CHECK(cudaMalloc(&d_L, vec_size));

    // --- Copy inputs to device ---
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, mat_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, mat_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, mat_size, cudaMemcpyHostToDevice));

    // --- Launch kernel with CUDA event timing ---
    //   Grid:  (N / Br) blocks — one per Q-tile
    //   Block: (Bc, Br) threads — col × row within the tile
    dim3 grid(N / Br);
    dim3 block(Bc, Br);

    printf("Launching kernel: grid=(%d), block=(%d, %d)\n", N / Br, Bc, Br);

    // =====================================================================
    // CUDA Events — the correct way to time GPU kernels
    // =====================================================================
    // Why not just use CPU timers (gettimeofday, clock(), etc.)?
    //   CPU timers measure wall time including launch overhead and
    //   synchronization latency. CUDA events are recorded directly on the
    //   GPU timeline, giving you the true kernel execution time.
    //
    // Workflow:
    //   1. Create two events (start, stop)
    //   2. Record start event into the stream  (zero-cost GPU timestamp)
    //   3. Launch kernel
    //   4. Record stop event into the stream
    //   5. Synchronize (wait for stop event to complete)
    //   6. Query elapsed time between the two events
    //   7. Destroy events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Warmup run (optional but good practice) ---
    // The first kernel launch often includes one-time costs:
    //   - JIT compilation of PTX → SASS (if not pre-compiled for this arch)
    //   - Context initialization, memory page mapping
    // A warmup run absorbs these so the timed run reflects steady-state.
    flash_attention_forward<<<grid, block>>>(d_Q, d_K, d_V, d_O, d_L, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Timed run ---
    CUDA_CHECK(cudaEventRecord(start));  // GPU timestamp: start

    flash_attention_forward<<<grid, block>>>(d_Q, d_K, d_V, d_O, d_L, N);

    CUDA_CHECK(cudaEventRecord(stop));   // GPU timestamp: stop
    CUDA_CHECK(cudaEventSynchronize(stop));  // block CPU until stop event fires

    // --- Compute elapsed time ---
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // --- Performance metrics ---
    // Total FLOPs for attention (forward only):
    //   Score matmul:    2 * N * N * d       (each dot product = 2d FLOPs)
    //   Softmax:         ~5 * N * N          (exp, sub, div per element)
    //   Output matmul:   2 * N * N * d
    //   Total ≈          4 * N * N * d
    double total_flops = 4.0 * N * N * d;
    double tflops = (total_flops / (elapsed_ms * 1e-3)) / 1e12;

    printf("\n--- Kernel Timing (CUDA Events) ---\n");
    printf("  Elapsed time:  %.4f ms\n", elapsed_ms);
    printf("  Approx FLOPs:  %.2e\n", total_flops);
    printf("  Throughput:    %.4f TFLOP/s\n", tflops);

    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaGetLastError());

    // --- Copy output back ---
    CUDA_CHECK(cudaMemcpy(h_O, d_O, mat_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_L, d_L, vec_size, cudaMemcpyDeviceToHost));

    // --- Compute reference on CPU ---
    printf("Computing naive attention on CPU for verification...\n");
    naive_attention_cpu(h_Q, h_K, h_V, h_O_ref, N, d);

    // --- Verify correctness ---
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    for (int i = 0; i < N * d; i++) {
        float diff = fabsf(h_O[i] - h_O_ref[i]);
        if (diff > max_diff) max_diff = diff;
        avg_diff += diff;
    }
    avg_diff /= (N * d);

    printf("\nVerification against naive attention:\n");
    printf("  Max absolute error: %.6e\n", max_diff);
    printf("  Avg absolute error: %.6e\n", avg_diff);

    if (max_diff < 1e-3f) {
        printf("  PASSED — Flash Attention matches naive implementation\n");
    } else {
        printf("  FAILED — errors too large, check implementation\n");
    }

    // --- Cleanup ---
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_O_ref); free(h_L);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);

    return 0;
}
