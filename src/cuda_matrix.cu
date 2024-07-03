#include "cuda_matrix.cuh"

template<unsigned int TILE_DIM>
__global__ void rowColProduct(float* dataA, float* dataB, float* dataC, unsigned int N){

	// This shared memory exists in each block of threads 
	__shared__ float sharedA[TILE_DIM][TILE_DIM];
	__shared__ float sharedB[TILE_DIM][TILE_DIM];

	// Save ids into registers for FAST access
	unsigned int blockCol = blockIdx.x;
	unsigned int blockRow = blockIdx.y;
	unsigned int tileCol = threadIdx.x;
	unsigned int tileRow = threadIdx.y;

	// Deduce row and columns based on the position of the thread
	unsigned int col = blockCol * TILE_DIM + tileCol;
	unsigned int row = blockRow * TILE_DIM + tileRow;

	float dot = 0;

	// The value N/static_cast<float>(TILE_DIM) is the number of tiles that will need to be computed
	for(unsigned int p = 0; p < N/static_cast<float>(TILE_DIM); ++p){
		// Populate the shared memory
		sharedA[tileRow][tileCol] = dataA[row * N + p * TILE_DIM + tileCol];
		sharedB[tileRow][tileCol] = dataB[(p * TILE_DIM + tileRow) * N + col];

		// Only proceed once shared memory has been populated
		__syncthreads();

		for(unsigned int i = 0; i < N; ++i){
			dot += sharedA[tileRow][i] * sharedB[i][tileCol];
		}

		// This prevents newer kernels from overwriting the shared memory that other kernels are done using yet
		__syncthreads();
	}

	dataC[row * N + col] = dot;
}

cudaMatrix::cudaMatrix(unsigned int N, float* data) : mN{N}, mData{nullptr} {
	if(cudaError_t err = cudaMalloc(&mData, mN * mN * sizeof(float)); err != cudaSuccess) std::cout << cudaGetErrorString(err);
	cudaMemcpy(mData, data, mN * mN * sizeof(float), cudaMemcpyHostToDevice);
}

void cudaMatrix::syncHost(float* hostData){
	cudaMemcpy(hostData, mData, mN * mN * sizeof(float), cudaMemcpyDeviceToHost);
}

void cudaMatrix::multiply(cudaMatrix &matA, cudaMatrix &matB, cudaMatrix &matC){

	constexpr unsigned int BLOCK_DIM = 32;
	constexpr dim3 blockDimension{BLOCK_DIM, BLOCK_DIM};
	dim3 gridDimension{static_cast<unsigned int>(std::ceil(static_cast<float>(matA.mN)/BLOCK_DIM)), static_cast<unsigned int>(std::ceil(static_cast<float>(matA.mN)/BLOCK_DIM))};

	if(matA.mN != matB.mN || matA.mN != matC.mN) throw std::runtime_error("Matrices are not the same size!");

	rowColProduct<BLOCK_DIM><<<gridDimension, blockDimension>>>(matA.mData, matB.mData, matC.mData, matA.mN);
}

cudaMatrix::~cudaMatrix(){
	cudaFree(mData);
}
