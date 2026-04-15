/*
 * reduction_nvidia.cu
 * Parallel Reduction - NVIDIA A100 (CUDA)
 *
 * Three kernels:
 *   K1: reduce_baseline  - naive interleaved addressing (thread divergence)
 *   K2: reduce_shared    - sequential addressing + shared memory + 2-elem load
 *   K3: reduce_warp      - warp shuffle (__shfl_down_sync, warpSize=32) + loop unroll
 *
 * Run:
 *   ./reduction_nvidia [array_size]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/* Macros & timing                                                    */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

static cudaEvent_t t0, t1;
static void   timer_start(void) { cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventRecord(t0); }
static float  timer_stop (void) {
    float ms; cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1); return ms;
}

/* ================================================================== */
/* K1 - Baseline: interleaved addressing, heavy thread divergence     */
/* ================================================================== */
__global__ void reduce_baseline(const float *g_in, float *g_out, int n)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n) ? g_in[i] : 0.0f;
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)          /* divergent branch */
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

/* ================================================================== */
/* K2 - Shared memory optimized                                       */
/* ================================================================== */
__global__ void reduce_shared(const float *g_in, float *g_out, int n)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + tid;

    /* Load two elements — keeps all threads busy at the first step */
    float v = (i < n) ? g_in[i] : 0.0f;
    if (i + blockDim.x < n) v += g_in[i + blockDim.x];

    sdata[tid] = v;
    __syncthreads();

    /* Sequential stride, no divergence */
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    /* Final warp: unroll completely (warp executes in lockstep) */
    if (tid < 32) {
        volatile float *vs = sdata;
        vs[tid] += vs[tid + 32];
        vs[tid] += vs[tid + 16];
        vs[tid] += vs[tid +  8];
        vs[tid] += vs[tid +  4];
        vs[tid] += vs[tid +  2];
        vs[tid] += vs[tid +  1];
    }

    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

/* ================================================================== */
/* K3 - Warp shuffle optimized (NVIDIA-specific, warpSize = 32)       */
/* ================================================================== */
__global__ void reduce_warp(const float *g_in, float *g_out, int n)
{
    __shared__ float warp_sums[32];   /* one slot per warp; max 1024/32 = 32 */

    unsigned int tid    = threadIdx.x;
    unsigned int laneId = tid & 31;
    unsigned int warpId = tid >> 5;
    unsigned int i      = blockIdx.x * (blockDim.x * 2) + tid;

    /* Load two elements */
    float val = (i < n) ? g_in[i] : 0.0f;
    if (i + blockDim.x < n) val += g_in[i + blockDim.x];

    /* Within-warp reduction via shuffle (register-to-register, no smem) */
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);

    /* Lane 0 of each warp stores partial sum */
    if (laneId == 0) warp_sums[warpId] = val;
    __syncthreads();

    /* First warp reduces all partial sums */
    unsigned int num_warps = (blockDim.x + 31) / 32;
    val = (tid < num_warps) ? warp_sums[tid] : 0.0f;
    if (warpId == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    if (tid == 0) g_out[blockIdx.x] = val;
}

/* ================================================================== */
/* Host driver: two-pass reduction                                    */
/* ================================================================== */
static float run_kernel(int kid, const float *d_in, float *d_tmp, float *d_out,
                         int n, int threads, float *out_ms)
{
    int smem = threads * sizeof(float);

    timer_start();

    int curr_n = n;

    float *d_src = (float *)d_in;
    float *d_a   = d_tmp;
    float *d_b   = d_out;
    float *d_dst = d_a;

    while (curr_n > 1) {
        int blocks = (kid == 1)
                   ? (curr_n + threads - 1) / threads
                   : (curr_n + threads * 2 - 1) / (threads * 2);

        if (kid == 1)
            reduce_baseline<<<blocks, threads, smem>>>(d_src, d_dst, curr_n);
        else if (kid == 2)
            reduce_shared<<<blocks, threads, smem>>>(d_src, d_dst, curr_n);
        else
            reduce_warp<<<blocks, threads, 32 * sizeof(float)>>>(d_src, d_dst, curr_n);

        CUDA_CHECK(cudaDeviceSynchronize());

        curr_n = blocks;

        /* Keep d_in read-only across runs by ping-ponging only workspace buffers. */
        d_src = d_dst;
        d_dst = (d_dst == d_a) ? d_b : d_a;
    }

    *out_ms = timer_stop();

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, d_src, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

/* ================================================================== */
/* Main                                                               */
/* ================================================================== */
int main(int argc, char *argv[])
{
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);
    int max_n = 1 << 26; 
    int threads = 256;

    printf("==========================================================\n");
    printf("  Parallel Reduction - NVIDIA A100 (CUDA)\n");
    printf("  Array: %d elements  (%.2f MB)\n", n, n*4.0/1048576.0);
    printf("==========================================================\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float peak_bw = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6f;
    printf("Device: %s | SMs: %d | WarpSize: %d | Mem: %.1f GB | Peak BW: %.1f GB/s\n\n",
           prop.name, prop.multiProcessorCount, prop.warpSize,
           prop.totalGlobalMem / 1073741824.0, peak_bw);

    /* Host data */
    // float *h = (float *)malloc(n * sizeof(float));
    // for (int i = 0; i < n; i++) h[i] = 1.0f;
    float *h = (float *)malloc(max_n * sizeof(float));
    for (int i = 0; i < max_n; i++) h[i] = 1.0f;
    float expected = (float)n;

    /* Device buffers */
    float *d_in, *d_tmp, *d_out;
    int work_elems = (n + threads - 1) / threads; /* worst-case first pass is K1 */
    CUDA_CHECK(cudaMalloc(&d_in,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp, work_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, work_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h, n * sizeof(float), cudaMemcpyHostToDevice));

    int warmup     = 3;
    int bench      = 10;
    float base_ms  = 0.0f;

    const char *names[] = {
        "K1: Baseline (interleaved, divergent)",
        "K2: Shared Mem Opt (sequential+unroll)",
        "K3: Warp Shuffle (__shfl_down_sync)"
    };

    /* Main results CSV */
    FILE *csv = fopen("results_nvidia.csv", "w");
    fprintf(csv, "kernel_id,kernel_name,array_size,threads,time_ms,"
                 "bandwidth_GBs,pct_peak_bw,speedup_vs_baseline,correct\n");

    for (int kid = 1; kid <= 3; kid++) {
        for (int w = 0; w < warmup; w++) { float ms; run_kernel(kid, d_in, d_tmp, d_out, n, threads, &ms); }

        double tot = 0.0; float result = 0.0f;
        for (int r = 0; r < bench; r++) {
            float ms;
            result = run_kernel(kid, d_in, d_tmp, d_out, n, threads, &ms);
            tot += ms;
        }

        float avg_ms  = (float)(tot / bench);
        float bw      = (n * 4.0f) / (avg_ms * 1e-3f) / 1e9f;
        float pct     = bw / peak_bw * 100.0f;
        float speedup = (kid == 1) ? 1.0f : base_ms / avg_ms;
        float rel_err = fabsf(result - expected) / expected;
        int ok = rel_err < 1e-5f;
        if (kid == 1) base_ms = avg_ms;

        printf("--- %s\n  Time: %.4f ms | BW: %.2f GB/s | %%Peak: %.1f%% | Speedup: %.2fx [%s]\n\n",
               names[kid-1], avg_ms, bw, pct, speedup, ok ? "OK" : "FAIL");
        fprintf(csv, "%d,\"%s\",%d,%d,%.4f,%.4f,%.2f,%.4f,%d\n",
                kid, names[kid-1], n, threads, avg_ms, bw, pct, speedup, ok);
    }
    fclose(csv);

    /* Sweep CSV */
    FILE *sweep = fopen("sweep_nvidia.csv", "w");
    fprintf(sweep, "array_size,kernel_id,time_ms,bandwidth_GBs\n");

    int sizes[] = { 1<<16, 1<<18, 1<<20, 1<<22, 1<<24, 1<<26 };
    for (int si = 0; si < 6; si++) {
        int sz = sizes[si];
        float *d_i2, *d_t2, *d_o2;
        int work_elems2 = (sz + threads - 1) / threads;
        CUDA_CHECK(cudaMalloc(&d_i2, sz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_t2, work_elems2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_o2, work_elems2 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_i2, h, sz * sizeof(float), cudaMemcpyHostToDevice));

        for (int kid = 1; kid <= 3; kid++) {
            double tot = 0.0; float ms;
            for (int r = 0; r < 5; r++) { run_kernel(kid, d_i2, d_t2, d_o2, sz, threads, &ms); tot += ms; }
            float avg = (float)(tot / 5);
            fprintf(sweep, "%d,%d,%.4f,%.4f\n", sz, kid, avg, (sz*4.0f)/(avg*1e-3f)/1e9f);
        }
        cudaFree(d_i2); cudaFree(d_t2); cudaFree(d_o2);
    }
    fclose(sweep);

    printf("Results written to results_nvidia.csv and sweep_nvidia.csv\n");
    cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out);
    free(h);
    return 0;
}
