#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdlib>

#define NUM_RUNS 500
#define N 1024

#if UF > 0
    #define PRAGMA_EXPAND(x) _Pragma(#x)
    #define UNROLL(x) PRAGMA_EXPAND(unroll x)
#else
    #define UNROLL(x)
#endif

// Tuning macros injected with compiler flags (for the auto python script)
#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 16
#endif
#define BLOCK_SIZE_Y BLOCK_SIZE_X

#ifndef REG_TILE_X
#define REG_TILE_X 2
#endif
#define REG_TILE_Y REG_TILE_X

#ifndef VEC_TYPE
#define VEC_TYPE float4 
#endif

#ifndef UF
#define UF 0 
#endif

#ifndef NUM_STREAMS
#define NUM_STREAMS 4
#endif

#define TILE_DIM (BLOCK_SIZE_X * REG_TILE_X)

#define SMEM_PADDING (((BLOCK_SIZE_X) - ((TILE_DIM) % 32) + 32) % 32)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

//Naive
__global__ void gemm_v0(const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        #pragma unroll 1
        for (int k = 0; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

//Smem
__global__ void gemm_v1(const float* A, const float* B, float* C) {
    __shared__ float sA[TILE_DIM][TILE_DIM + SMEM_PADDING];
    __shared__ float sB[TILE_DIM][TILE_DIM + SMEM_PADDING];
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
                sB[ty + i * BLOCK_SIZE_Y][tx + j * BLOCK_SIZE_X] = (rB < N && cB < N) ? B[rB * N + cB] : 0.0f;
            }
        }
        __syncthreads();
        #pragma unroll 1
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

//Unroll control
__global__ void gemm_v2_unrolled(const float* A, const float* B, float* C) {
    __shared__ float sA[TILE_DIM][TILE_DIM + SMEM_PADDING];
    __shared__ float sB[TILE_DIM][TILE_DIM + SMEM_PADDING];
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
                sB[ty + i * BLOCK_SIZE_Y][tx + j * BLOCK_SIZE_X] = (rB < N && cB < N) ? B[rB * N + cB] : 0.0f;
            }
        }
        __syncthreads();
        UNROLL(UF)
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

//Vectorization
__global__ void gemm_v3_vectorized(const float* A, const float* B, float* C) {
    __shared__ float sA[TILE_DIM][TILE_DIM + SMEM_PADDING];
    __shared__ float sB[TILE_DIM][TILE_DIM + SMEM_PADDING];
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x;  int by = blockIdx.y;
    int row_start = by * TILE_DIM + ty;
    int col_start = bx * TILE_DIM + tx;
    float reg_C[REG_TILE_Y][REG_TILE_X] = {0.0f};
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;
    
    //This part is to check if the smem padding aligns with the vectorization size
    constexpr int vec_len = sizeof(VEC_TYPE) / sizeof(float);
    constexpr bool IS_SMEM_ALIGNED = ((TILE_DIM + SMEM_PADDING) % vec_len) == 0;
    constexpr int vecs_per_row = TILE_DIM / vec_len;
    constexpr int total_vecs = (TILE_DIM * TILE_DIM) / vec_len;

    for (int tile_idx = 0; tile_idx < (N + TILE_DIM - 1) / TILE_DIM; ++tile_idx) {
        for (int iter = tid; iter < total_vecs; iter += num_threads) {
            int tile_row = iter / vecs_per_row;
            int tile_col = (iter % vecs_per_row) * vec_len;
            int global_rA = by * TILE_DIM + tile_row;
            int global_cA = tile_idx * TILE_DIM + tile_col;
            int idx_A = global_rA * N + global_cA;
            
            if (global_rA < N && global_cA + vec_len <= N && (idx_A % vec_len == 0)) {
                if (IS_SMEM_ALIGNED) {
                    *reinterpret_cast<VEC_TYPE*>(&sA[tile_row][tile_col]) = *reinterpret_cast<const VEC_TYPE*>(&A[idx_A]);
                } else {
                    VEC_TYPE val = *reinterpret_cast<const VEC_TYPE*>(&A[idx_A]);
                    float* val_ptr = reinterpret_cast<float*>(&val);
                    for(int v = 0; v < vec_len; v++) sA[tile_row][tile_col + v] = val_ptr[v];
                }
            } else {
                for(int v = 0; v < vec_len; v++) sA[tile_row][tile_col + v] = (global_rA < N && global_cA + v < N) ? A[global_rA * N + global_cA + v] : 0.0f;
            }
        }
        for (int iter = tid; iter < total_vecs; iter += num_threads) {
            int tile_row = iter / vecs_per_row;
            int tile_col = (iter % vecs_per_row) * vec_len;
            int global_rB = tile_idx * TILE_DIM + tile_row;
            int global_cB = bx * TILE_DIM + tile_col;
            int idx_B = global_rB * N + global_cB;
            
            if (global_rB < N && global_cB + vec_len <= N && (idx_B % vec_len == 0)) {
                if (IS_SMEM_ALIGNED) {
                    *reinterpret_cast<VEC_TYPE*>(&sB[tile_row][tile_col]) = *reinterpret_cast<const VEC_TYPE*>(&B[idx_B]);
                } else {
                    VEC_TYPE val = *reinterpret_cast<const VEC_TYPE*>(&B[idx_B]);
                    float* val_ptr = reinterpret_cast<float*>(&val);
                    for(int v = 0; v < vec_len; v++) sB[tile_row][tile_col + v] = val_ptr[v];
                }
            } else {
                for(int v = 0; v < vec_len; v++) sB[tile_row][tile_col + v] = (global_rB < N && global_cB + v < N) ? B[global_rB * N + global_cB + v] : 0.0f;
            }
        }
        __syncthreads();
        UNROLL(UF)
        for (int k = 0; k < TILE_DIM; ++k) {
            #pragma unroll
            for (int i = 0; i < REG_TILE_Y; ++i) {
                #pragma unroll
                for (int j = 0; j < REG_TILE_X; ++j) reg_C[i][j] += sA[ty + i * BLOCK_SIZE_Y][k] * sB[k][tx + j * BLOCK_SIZE_X];
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

void init_matrix(float* mat, int size, int seed) {
    srand(seed);
    for (int i = 0; i < size; ++i) mat[i] = static_cast<float>(rand()) / RAND_MAX;
}


float run_batch_gemm(int version, float lastMilliseconds, float naiveMilliseconds) {
    size_t mat_elements = N * N;
    size_t mat_bytes = mat_elements * sizeof(float);
    
    float *h_A = new float[mat_elements], *h_B = new float[mat_elements], *h_C_batch;
    CUDA_CHECK(cudaMallocHost(&h_C_batch, NUM_RUNS * mat_bytes));
    init_matrix(h_A, mat_elements, 42); init_matrix(h_B, mat_elements, 42);
    
    float *d_A, *d_B, *d_C_batch;
    CUDA_CHECK(cudaMalloc(&d_A, mat_bytes)); CUDA_CHECK(cudaMalloc(&d_B, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_C_batch, NUM_RUNS * mat_bytes));
    CUDA_CHECK(cudaMemset(d_C_batch, 0, NUM_RUNS * mat_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, mat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, mat_bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    if (version == 0)
         grid = dim3((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (N + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Starting benchmark for Version " << version << "...\n";
    CUDA_CHECK(cudaEventRecord(start));

    bool out_of_resources = false;
    float milliseconds = 0;

    if (version < 4) {
        for(int i = 0; i < NUM_RUNS; i++) {
            float* h_C_current = h_C_batch + (i * mat_elements);
            float* d_C_current = d_C_batch + (i * mat_elements);
            
            if (version == 0) {
                if (naiveMilliseconds != 0) milliseconds = naiveMilliseconds;
                else gemm_v0<<<grid, block>>>(d_A, d_B, d_C_current);
            }
            else if (version == 1) gemm_v1<<<grid, block>>>(d_A, d_B, d_C_current);
            else if (version == 2) {
                if (UF == 1) milliseconds = lastMilliseconds;
                else gemm_v2_unrolled<<<grid, block>>>(d_A, d_B, d_C_current);
            }
            else if (version == 3) {
                if (sizeof(VEC_TYPE)/sizeof(float) == 1) milliseconds = lastMilliseconds;
                else gemm_v3_vectorized<<<grid, block>>>(d_A, d_B, d_C_current);
            }

            CUDA_CHECK(cudaMemcpy(h_C_current, d_C_current, mat_bytes, cudaMemcpyDeviceToHost));
        }
    } 
    else if (version == 4) {

        if (NUM_STREAMS == 1) {
            milliseconds = lastMilliseconds;
        }

        else {
            cudaStream_t streams[NUM_STREAMS];
            for (int i = 0; i < NUM_STREAMS; ++i) {
                    CUDA_CHECK(cudaStreamCreate(&streams[i]));
            }

            for(int i = 0; i < NUM_RUNS; i++) {
                    int stream_idx = i % NUM_STREAMS;
                    float* h_C_current = h_C_batch + (i * mat_elements);
                    float* d_C_current = d_C_batch + (i * mat_elements);
                    gemm_v3_vectorized<<<grid, block, 0, streams[stream_idx]>>>(d_A, d_B, d_C_current);
                    CUDA_CHECK(cudaMemcpyAsync(h_C_current, d_C_current, mat_bytes, cudaMemcpyDeviceToHost, streams[stream_idx]));
            }

            for (int i = 0; i < NUM_STREAMS; ++i) {
                    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
                    CUDA_CHECK(cudaStreamDestroy(streams[i]));
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    if (milliseconds == 0.0f)
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Total Time: " << milliseconds << " ms\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_batch); cudaFreeHost(h_C_batch);
    delete[] h_A; delete[] h_B; cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return milliseconds;
}

int main() {
    float lastMilliseconds = 0;
    float naiveMilliseconds = run_batch_gemm(0, 0, 0);
    for (int v = 0; v < 5; ++v) {
        lastMilliseconds = run_batch_gemm(v, lastMilliseconds, naiveMilliseconds);
        if (lastMilliseconds < 0.0f) {
            std::cout << "Skipping remaining versions due to resource exhaustion.\n";
            break; 
        }
    }
    return 0;
}