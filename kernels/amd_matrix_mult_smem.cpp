#include <iostream>
#include <hip/hip_runtime.h>
#define N 20
#define BLOCK_SIZE 8


__global__ void multMatrices(const float (*A)[N], const float (*B)[N], float (*C)[N]) {

	// Shared memory optimization
	__shared__ float rowA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float colB[BLOCK_SIZE][BLOCK_SIZE];

	// Calculate global column and row separately
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	// Sum for totalling the products of row/column pairs from A and B to calculate each element of C
	float sum = 0.0f;

	// Since we load only BLOCK_SIZE elements from a row (from A) or column (from B) at a time, we must loop ceil(N/BLOCK_SIZE) times
	for(int n = 0 ; n < (N + BLOCK_SIZE-1)/BLOCK_SIZE ; n++) {

		// Calculating this iteration's tile coordinates for each thread
		int tileCol = threadIdx.x + n*blockDim.x;
		int tileRow = threadIdx.y + n*blockDim.y;

		// Loading it the tile temporarily (if the coordinate is not out of bounds of A and B)
		rowA[threadIdx.y][threadIdx.x] = (row < N && tileCol < N) ? A[row][tileCol] : 0.0f;
		colB[threadIdx.y][threadIdx.x] = (tileRow < N && col < N) ? B[tileRow][col] : 0.0f;

		// Making sure all temporary tile elements are loaded
		__syncthreads();

		// Calculating the product of row/column pairs for this tile and adding it to the total sum for all tiles
		#pragma unroll
		for(int i = 0 ; i < BLOCK_SIZE ; i++) {

			sum += rowA[threadIdx.y][i] * colB[i][threadIdx.x];

		}

		// Syncing again otherwise other warps may overwrite the temporary tile in the next iteration
		__syncthreads();

	}

	if(row < N && col < N)
		C[row][col] = sum;	

}



void printMatrix(const float (*X)[N], int length) {

	for (int y = 0 ; y < N ; y++) {

		for (int x = 0 ; x < length ; x++) {

			std::cout << X[y][x] << " ";

		}
		std::cout << "\n";

	}

}


int main() {


	// Allocate host memory
	float (*h_A)[N] = new float[N][N];
	float (*h_B)[N] = new float[N][N];
	float (*h_C)[N] = new float[N][N]();


	// Initialize input vectors
	for(int i = 0 ; i < N ; i++) {
		for (int j = 0 ; j < N ; j++) {

			h_A[i][j] = 100.0f * ((float) rand() / (float) RAND_MAX);
			h_B[i][j] = 100.0f * ((float) rand() / (float) RAND_MAX);

		}

	}


	// Allocate device memory
	float (*d_A)[N], (*d_B)[N], (*d_C)[N];

	size_t size = N * N * sizeof(float);

	hipMalloc((void**)&d_A, size);
	hipMalloc((void**)&d_B, size);
	hipMalloc((void**)&d_C, size);

	hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
	hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

	hipEvent_t start, stop;

	hipEventCreate(&start);
	hipEventCreate(&stop);

	dim3 numThreadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks( ((N + BLOCK_SIZE - 1) / BLOCK_SIZE),
			((N + BLOCK_SIZE - 1) / BLOCK_SIZE) );
	
	hipEventRecord(start);
	multMatrices<<<numBlocks, numThreadsPerBlock>>>(d_A, d_B, d_C);

	hipEventRecord(stop);
	hipEventSynchronize(stop);

	float ms = 0;
	hipEventElapsedTime(&ms, start, stop);
	float FLOPs = ((float) N / ms) * 1000.0f;
	printf("Threads per block = %d , Elapsed time = %fms , FLOPS = %f\n", numThreadsPerBlock, ms, FLOPs);

	hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);

	printMatrix(h_C, N);

}
