#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdlib>

// --- Experiment parameters ---
#define NUM_RUNS 500
#define N 1024

// --- MACROS FOR TUNING ---
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define REG_TILE_X 2
#define REG_TILE_Y 2
#define TILE_DIM (BLOCK_SIZE_X * REG_TILE_X)

#define UF 4
const int UNROLL_FACTOR = UF;
#define NUM_STREAMS 4
#define VEC_TYPE float4 // Can be float, float2, or float4

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ==============================================================================
// V0: Naive GEMM
// ==============================================================================
__global__ void gemm_v0(const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ==============================================================================
// V1: Shared Memory & Register Tiling
// ==============================================================================
__global__ void gemm_v1(const float* A, const float* B, float* C) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x;  int by = blockIdx.y;

    int row_start = by * TILE_DIM + ty;
    int col_start = bx * TILE_DIM + tx;

    float reg_C[REG_TILE_Y][REG_TILE_X] = {0.0f};

    for (int tile_idx = 0; tile_idx < (N + TILE_DIM - 1) / TILE_DIM; ++tile_idx) {
        // Collaborative load into shared memory (simplified for 1-to-1 mapping here)
        for(int i = 0; i < REG_TILE_Y; i++) {
            for(int j = 0; j < REG_TILE_X; j++) {
                int r = row_start + i * BLOCK_SIZE_Y;
                int c = tile_idx * TILE_DIM + tx + j * BLOCK_SIZE_X;
                sA[ty + i * BLOCK_SIZE_Y][tx + j * BLOCK_SIZE_X] = (r < N && c < N) ? A[r * N + c] : 0.0f;
                
                r = tile_idx * TILE_DIM + ty + i * BLOCK_SIZE_Y;
                c = col_start + j * BLOCK_SIZE_X;
                sB[ty + i * BLOCK_SIZE_Y][tx + j * BLOCK_SIZE_X] = (r < N && c < N) ? B[r * N + c] : 0.0f;
            }
        }
        __syncthreads();

        // Compute from shared memory into registers
        for (int k = 0; k < TILE_DIM; ++k) {
            for (int i = 0; i < REG_TILE_Y; ++i) {
                for (int j = 0; j < REG_TILE_X; ++j) {
                    reg_C[i][j] += sA[ty + i * BLOCK_SIZE_Y][k] * sB[k][tx + j * BLOCK_SIZE_X];
                }
            }
        }
        __syncthreads();
    }

    // Write back to global memory
    for (int i = 0; i < REG_TILE_Y; ++i) {
        for (int j = 0; j < REG_TILE_X; ++j) {
            int r = row_start + i * BLOCK_SIZE_Y;
            int c = col_start + j * BLOCK_SIZE_X;
            if (r < N && c < N) C[r * N + c] = reg_C[i][j];
        }
    }
}


// ==============================================================================
// V2: V1 + Loop Unrolling
// ==============================================================================
__global__ void gemm_v2_unrolled(const float* A, const float* B, float* C) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x;  int by = blockIdx.y;

    int row_start = by * TILE_DIM + ty;
    int col_start = bx * TILE_DIM + tx;

    float reg_C[REG_TILE_Y][REG_TILE_X] = {0.0f};

    for (int tile_idx = 0; tile_idx < (N + TILE_DIM - 1) / TILE_DIM; ++tile_idx) {
        for(int i = 0; i < REG_TILE_Y; i++) {
            for(int j = 0; j < REG_TILE_X; j++) {
                int rA = row_start + i * BLOCK_SIZE_Y;
                int cA = tile_idx * TILE_DIM + tx + j * BLOCK_SIZE_X;
                sA[ty + i * BLOCK_SIZE_Y][tx + j * BLOCK_SIZE_X] = (rA < N && cA < N) ? A[rA * N + cA] : 0.0f;
                
                int cB = col_start + j * BLOCK_SIZE_X; 
                int rB = tile_idx * TILE_DIM + ty + i * BLOCK_SIZE_Y; 
                sB[ty + i * BLOCK_SIZE_Y][tx + j * BLOCK_SIZE_X] = (rB < N && cB < N) ? B[cB * N + rB] : 0.0f;
            }
        }
        __syncthreads();

        // The pragma uses your macro to unroll the computation loop
        #pragma unroll UNROLL_FACTOR
        for (int k = 0; k < TILE_DIM; ++k) {
            for (int i = 0; i < REG_TILE_Y; ++i) {
                for (int j = 0; j < REG_TILE_X; ++j) {
                    reg_C[i][j] += sA[ty + i * BLOCK_SIZE_Y][k] * sB[k][tx + j * BLOCK_SIZE_X];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < REG_TILE_Y; ++i) {
        for (int j = 0; j < REG_TILE_X; ++j) {
            int r = row_start + i * BLOCK_SIZE_Y;
            int c = col_start + j * BLOCK_SIZE_X;
            if (r < N && c < N) C[r * N + c] = reg_C[i][j];
        }
    }
}

// ==============================================================================
// V3: V2 + Vectorization (Using VEC_TYPE)
// ==============================================================================
__global__ void gemm_v3_vectorized(const float* A, const float* B, float* C) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x;  int by = blockIdx.y;

    int row_start = by * TILE_DIM + ty;
    int col_start = bx * TILE_DIM + tx;

    float reg_C[REG_TILE_Y][REG_TILE_X] = {0.0f};

    // Calculate how many floats are in the chosen vector type
    constexpr int vec_len = sizeof(VEC_TYPE) / sizeof(float);
    
    // Cast global memory pointers to vector types
    const VEC_TYPE* vec_A = reinterpret_cast<const VEC_TYPE*>(A);
    const VEC_TYPE* vec_B = reinterpret_cast<const VEC_TYPE*>(B);

    for (int tile_idx = 0; tile_idx < (N + TILE_DIM - 1) / TILE_DIM; ++tile_idx) {
        // Adjust the inner loop to increment by vector length
        for(int i = 0; i < REG_TILE_Y; i++) {
            for(int j = 0; j < REG_TILE_X; j += vec_len) {
                
                int rA = row_start + i * BLOCK_SIZE_Y;
                int cA = tile_idx * TILE_DIM + tx + j * BLOCK_SIZE_X;
                
                // Vectorized load for A
                if (rA < N && cA < N) {
                    VEC_TYPE val_A = vec_A[(rA * N + cA) / vec_len];
                    float* val_A_ptr = reinterpret_cast<float*>(&val_A);
                    for (int v = 0; v < vec_len; ++v) {
                        sA[ty + i * BLOCK_SIZE_Y][tx + (j + v) * BLOCK_SIZE_X] = val_A_ptr[v];
                    }
                } else {
                    for (int v = 0; v < vec_len; ++v) {
                        sA[ty + i * BLOCK_SIZE_Y][tx + (j + v) * BLOCK_SIZE_X] = 0.0f;
                    }
                }
                
                // Vectorized load for B
                int cB = col_start + j * BLOCK_SIZE_X; 
                int rB = tile_idx * TILE_DIM + ty + i * BLOCK_SIZE_Y; 
                
                if (rB < N && cB < N) {
                    VEC_TYPE val_B = vec_B[(cB * N + rB) / vec_len];
                    float* val_B_ptr = reinterpret_cast<float*>(&val_B);
                    for (int v = 0; v < vec_len; ++v) {
                        sB[ty + i * BLOCK_SIZE_Y][tx + (j + v) * BLOCK_SIZE_X] = val_B_ptr[v];
                    }
                } else {
                    for (int v = 0; v < vec_len; ++v) {
                         sB[ty + i * BLOCK_SIZE_Y][tx + (j + v) * BLOCK_SIZE_X] = 0.0f;
                    }
                }
            }
        }
        __syncthreads();

        #pragma unroll UNROLL_FACTOR
        for (int k = 0; k < TILE_DIM; ++k) {
            for (int i = 0; i < REG_TILE_Y; ++i) {
                for (int j = 0; j < REG_TILE_X; ++j) {
                    reg_C[i][j] += sA[ty + i * BLOCK_SIZE_Y][k] * sB[k][tx + j * BLOCK_SIZE_X];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < REG_TILE_Y; ++i) {
        for (int j = 0; j < REG_TILE_X; ++j) {
            int r = row_start + i * BLOCK_SIZE_Y;
            int c = col_start + j * BLOCK_SIZE_X;
            if (r < N && c < N) C[r * N + c] = reg_C[i][j];
        }
    }
}


// Helper to initialize matrix with a fixed seed
void init_matrix(float* mat, int size, int seed) {
    srand(seed);
    for (int i = 0; i < size; ++i) {
        // Generate values between 0.0 and 1.0
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}


// ==============================================================================
// Updated Host Code
// ==============================================================================
void run_batch_gemm(int version) {
    size_t mat_elements = N * N;
    size_t mat_bytes = mat_elements * sizeof(float);
    
    // 1. Host allocation
    float* h_A = new float[mat_elements];
    float* h_B = new float[mat_elements];
    
    // 2. Fixed-seed initialization
    init_matrix(h_A, mat_elements, 42);
    init_matrix(h_B, mat_elements, 42);
    
    // 3. Device allocation
    float *d_A, *d_B, *d_C_batch;
    CUDA_CHECK(cudaMalloc(&d_A, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, mat_bytes));
    
    // Allocate a contiguous 4GB block for 1000 output matrices to prevent race conditions 
    // when streams write concurrently.
    CUDA_CHECK(cudaMalloc(&d_C_batch, NUM_RUNS * mat_bytes));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, mat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, mat_bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    // 4. Setup Timing Events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up to avoid cold-start latency skewing the benchmark
    gemm_v0<<<grid, block>>>(d_A, d_B, d_C_batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Starting benchmark for Version " << version << " with " << NUM_RUNS << " runs..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));

    if (version < 4) {
        // V0 - V3: Default stream, sequential kernel launches
        for(int i = 0; i < NUM_RUNS; i++) {
            float* d_C_current = d_C_batch + (i * mat_elements);
            
            if (version == 0) gemm_v0<<<grid, block>>>(d_A, d_B, d_C_current);
            else if (version == 1) gemm_v1<<<grid, block>>>(d_A, d_B, d_C_current);
            else if (version == 2) gemm_v2_unrolled<<<grid, block>>>(d_A, d_B, d_C_current);
            else if (version == 3) gemm_v3_vectorized<<<grid, block>>>(d_A, d_B, d_C_current);
        }
    } 
    else if (version == 4) {
        // V4: Stream parallelization using the fastest kernel (V3)
        cudaStream_t streams[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }

        // Round-robin distribution partitions the work as fairly as mathematically possible
        for(int i = 0; i < NUM_RUNS; i++) {
            int stream_idx = i % NUM_STREAMS;
            float* d_C_current = d_C_batch + (i * mat_elements);
            gemm_v3_vectorized<<<grid, block, 0, streams[stream_idx]>>>(d_A, d_B, d_C_current);
        }

        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Total Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Average Time per GEMM: " << (milliseconds / NUM_RUNS) << " ms" << std::endl;

    // 5. Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_batch);
    delete[] h_A; delete[] h_B;
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    
    // Loop through versions 0 to 4 to generate your benchmark table
    for (int v = 0; v < 5; ++v) {
        run_batch_gemm(v);
        std::cout << "----------------------------------------\n";
    }
    
    return 0;
}