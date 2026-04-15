/*
 * reduction_amd.cpp
 * Parallel Reduction - AMD MI50 (HIP)
 *
 * Three kernels:
 *   K1: reduce_baseline   - naive interleaved addressing (thread divergence)
 *   K2: reduce_shared     - sequential addressing + shared memory + 2-elem load
 *   K3: reduce_wavefront  - wavefront shuffle (__shfl_down, warpSize=64 on MI50)
 *
 * KEY AMD DIFFERENCES from the NVIDIA version:
 *   - wavefront size = 64 (GCN/gfx906), not 32
 *   - __shfl_down() has no sync mask parameter (AMD GCN is implicitly convergent)
 *   - Final unroll in K2 covers 64 levels, not 32
 *   - Shared mem for inter-warp in K3: blockDim.x/64 slots (not /32)
 *   - Block size must be multiple of 64; 256 recommended (avoids register spill)
 *
 * Compile:
 *   hipcc -O3 --offload-arch=gfx906 -o reduction_amd reduction_amd.cpp
 *
 * Run:
 *   ./reduction_amd [array_size]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>

/* ------------------------------------------------------------------ */
/* Macros & timing                                                      */
/* ------------------------------------------------------------------ */
#define HIP_CHECK(call)                                                 \
    do {                                                                \
        hipError_t err = (call);                                        \
        if (err != hipSuccess) {                                        \
            fprintf(stderr, "HIP error %s:%d  %s\n",                   \
                    __FILE__, __LINE__, hipGetErrorString(err));         \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

static hipEvent_t t0, t1;
static void  timer_start(void) { hipEventCreate(&t0); hipEventCreate(&t1); hipEventRecord(t0); }
static float timer_stop (void) {
    float ms; hipEventRecord(t1); hipEventSynchronize(t1);
    hipEventElapsedTime(&ms, t0, t1);
    hipEventDestroy(t0); hipEventDestroy(t1); return ms;
}

/* ================================================================== */
/* K1 - Baseline: interleaved addressing, heavy thread divergence      */
/*                                                                      */
/* On MI50 (wavefront=64) this is *worse* than on NVIDIA (warp=32):   */
/* 64 lanes must serialize divergent paths instead of 32.              */
/* ================================================================== */
__global__ void reduce_baseline(const float *g_in, float *g_out, int n)
{
    extern __shared__ float s[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + tid;

    s[tid] = (i < n) ? g_in[i] : 0.0f;
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)          /* divergent within 64-wide wavefront */
            s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = s[0];
}

/* ================================================================== */
/* K2 - Shared memory optimized                                        */
/*                                                                      */
/* Same logic as NVIDIA K2 but the final unroll covers 64 levels       */
/* (one full wavefront on GCN) rather than 32.                         */
/*                                                                      */
/* The volatile pointer trick avoids compiler reordering; no           */
/* __syncthreads needed because all 64 lanes execute together.         */
/* ================================================================== */
__global__ void reduce_shared(const float *g_in, float *g_out, int n)
{
    extern __shared__ float s[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + tid;

    float v = (i < n) ? g_in[i] : 0.0f;
    if (i + blockDim.x < n) v += g_in[i + blockDim.x];
    s[tid] = v;
    __syncthreads();

    /* Sequential stride, reduce to one wavefront (64 threads) */
    for (unsigned int stride = blockDim.x / 2; stride > 64; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    /* Final wavefront (64 threads): unroll completely, no sync needed */
    if (tid < 64) {
        volatile float *vs = s;
        vs[tid] += vs[tid + 64];   /* extra level vs NVIDIA: covers full wavefront */
        vs[tid] += vs[tid + 32];
        vs[tid] += vs[tid + 16];
        vs[tid] += vs[tid +  8];
        vs[tid] += vs[tid +  4];
        vs[tid] += vs[tid +  2];
        vs[tid] += vs[tid +  1];
    }
    if (tid == 0) g_out[blockIdx.x] = s[0];
}

/* ================================================================== */
/* K3 - Wavefront shuffle optimized (AMD MI50-specific)                */
/*                                                                      */
/* Key differences from the NVIDIA K3:                                 */
/*   - warpSize == 64 at runtime on MI50/GCN                          */
/*   - __shfl_down() has no mask (GCN wavefronts are always convergent)*/
/*   - Shuffle loop starts at offset 32 (half of 64), not 16           */
/*   - warp_sums[] needs blockDim.x/64 slots, not /32                 */
/*   - Block size should be a multiple of 64 (use 256)                 */
/* ================================================================== */
__global__ void reduce_wavefront(const float *g_in, float *g_out, int n)
{
    __shared__ float warp_sums[16];   /* max 1024/64 = 16 wavefronts */

    unsigned int tid    = threadIdx.x;
    unsigned int laneId = tid % warpSize;   /* 0..63 on MI50 */
    unsigned int warpId = tid / warpSize;
    unsigned int i      = blockIdx.x * (blockDim.x * 2) + tid;

    /* Load two elements */
    float val = (i < n) ? g_in[i] : 0.0f;
    if (i + blockDim.x < n) val += g_in[i + blockDim.x];

    /* Within-wavefront reduction via shuffle (register-to-register) */
    /* Offset starts at 32 = warpSize/2 = 64/2                       */
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
        val += __shfl_down(val, offset);   /* no mask on AMD */

    /* Lane 0 of each wavefront stores partial sum */
    if (laneId == 0) warp_sums[warpId] = val;
    __syncthreads();

    /* First wavefront reduces all partial sums */
    int numWarps = blockDim.x / warpSize;
    val = (tid < (unsigned int)numWarps) ? warp_sums[tid] : 0.0f;
    if (warpId == 0) {
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
    }
    if (tid == 0) g_out[blockIdx.x] = val;
}

/* ================================================================== */
/* Host driver: two-pass reduction                                      */
/* ================================================================== */
static float run_kernel(int kid, const float *d_in, float *d_tmp, float *d_out,
                         int n, int threads, float *out_ms)
{
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    int smem   = threads * sizeof(float);

    timer_start();

    if      (kid == 1) hipLaunchKernelGGL(reduce_baseline,   blocks, threads, smem, 0, d_in, d_tmp, n);
    else if (kid == 2) hipLaunchKernelGGL(reduce_shared,     blocks, threads, smem, 0, d_in, d_tmp, n);
    else               hipLaunchKernelGGL(reduce_wavefront,  blocks, threads, 16*sizeof(float), 0, d_in, d_tmp, n);

    if (blocks > 1) {
        int b2 = (blocks + threads * 2 - 1) / (threads * 2);
        if (b2 < 1) b2 = 1;
        if      (kid == 1) hipLaunchKernelGGL(reduce_baseline,  b2, threads, smem, 0, d_tmp, d_out, blocks);
        else if (kid == 2) hipLaunchKernelGGL(reduce_shared,    b2, threads, smem, 0, d_tmp, d_out, blocks);
        else               hipLaunchKernelGGL(reduce_wavefront, b2, threads, 16*sizeof(float), 0, d_tmp, d_out, blocks);
    } else {
        HIP_CHECK(hipMemcpy(d_out, d_tmp, sizeof(float), hipMemcpyDeviceToDevice));
    }

    HIP_CHECK(hipDeviceSynchronize());
    *out_ms = timer_stop();

    float result = 0.0f;
    HIP_CHECK(hipMemcpy(&result, d_out, sizeof(float), hipMemcpyDeviceToHost));
    return result;
}

/* ================================================================== */
/* Main                                                                 */
/* ================================================================== */
int main(int argc, char *argv[])
{
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);

    printf("==========================================================\n");
    printf("  Parallel Reduction - AMD MI50 (HIP)\n");
    printf("  Array: %d elements  (%.2f MB)\n", n, n*4.0/1048576.0);
    printf("==========================================================\n\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    float peak_bw = 1024.0f;   /* MI50 HBM2 spec: ~1024 GB/s */
    printf("Device: %s | CUs: %d | WavefrontSize: %d | Mem: %.1f GB | Peak BW: %.0f GB/s\n\n",
           prop.name, prop.multiProcessorCount, prop.warpSize,
           prop.totalGlobalMem / 1073741824.0, peak_bw);

    float *h = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h[i] = 1.0f;
    float expected = (float)n;

    float *d_in, *d_tmp, *d_out;
    HIP_CHECK(hipMalloc(&d_in,  n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tmp, ((n / 512) + 2) * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h, n * sizeof(float), hipMemcpyHostToDevice));

    int threads   = 256;   /* multiple of 64 (wavefront size); 256 avoids register spill on GCN */
    int warmup    = 3;
    int bench     = 10;
    float base_ms = 0.0f;

    const char *names[] = {
        "K1: Baseline (interleaved, divergent)",
        "K2: Shared Mem Opt (sequential+wavefront unroll)",
        "K3: Wavefront Shuffle (__shfl_down, AMD MI50)"
    };

    FILE *csv = fopen("results_amd.csv", "w");
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
        int   ok      = fabsf(result - expected) < 1.0f;
        if (kid == 1) base_ms = avg_ms;

        printf("--- %s\n  Time: %.4f ms | BW: %.2f GB/s | %%Peak: %.1f%% | Speedup: %.2fx [%s]\n\n",
               names[kid-1], avg_ms, bw, pct, speedup, ok ? "OK" : "FAIL");
        fprintf(csv, "%d,\"%s\",%d,%d,%.4f,%.4f,%.2f,%.4f,%d\n",
                kid, names[kid-1], n, threads, avg_ms, bw, pct, speedup, ok);
    }
    fclose(csv);

    FILE *sweep = fopen("sweep_amd.csv", "w");
    fprintf(sweep, "array_size,kernel_id,time_ms,bandwidth_GBs\n");

    int sizes[] = { 1<<16, 1<<18, 1<<20, 1<<22, 1<<24, 1<<26 };
    for (int si = 0; si < 6; si++) {
        int sz = sizes[si];
        float *d_i2, *d_t2, *d_o2;
        HIP_CHECK(hipMalloc(&d_i2, sz * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_t2, ((sz / 512) + 2) * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_o2, sizeof(float)));
        HIP_CHECK(hipMemcpy(d_i2, h, sz * sizeof(float), hipMemcpyHostToDevice));

        for (int kid = 1; kid <= 3; kid++) {
            double tot = 0.0; float ms;
            for (int r = 0; r < 5; r++) { run_kernel(kid, d_i2, d_t2, d_o2, sz, threads, &ms); tot += ms; }
            float avg = (float)(tot / 5);
            fprintf(sweep, "%d,%d,%.4f,%.4f\n", sz, kid, avg, (sz*4.0f)/(avg*1e-3f)/1e9f);
        }
        hipFree(d_i2); hipFree(d_t2); hipFree(d_o2);
    }
    fclose(sweep);

    printf("Results written to results_amd.csv and sweep_amd.csv\n");
    hipFree(d_in); hipFree(d_tmp); hipFree(d_out);
    free(h);
    return 0;
}
