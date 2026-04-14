#include <iostream>
#include <hip/hip_runtime.h>
#define N 20
#define BLOCK_SIZE 8


__global__ void multMatrices(const float (*A)[N], const float (*B)[N], float (*C)[N]) {

	int globalX = threadIdx.x + blockDim.x * blockIdx.x;
	int globalY = threadIdx.y + blockDim.y * blockIdx.y;
	
	if (globalX < N && globalY < N) {
		float sum = 0;

		for (int x = 0 ; x < N ; x++) {

			sum += A[globalY][x] * B[x][globalX];

		}

		C[globalY][globalX] = sum;
		
	}

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
