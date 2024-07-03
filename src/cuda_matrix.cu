#include "cuda_matrix.cuh"

__global__ void rowColProduct(double* dataA, double* dataB, double* dataC, unsigned int N){
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;

	unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;

	if(col < N && row < N){
		double sum = 0;

		for(unsigned int i = 0; i < N; ++i){
			sum += dataA[row * N + i] * dataB[col + N * i];
		}

		dataC[row * N + col] = sum;
	}
}

cudaMatrix::cudaMatrix(unsigned int N, double* data) : mN{N}, mData{nullptr} {
	if(cudaError_t err = cudaMalloc(&mData, mN * mN * sizeof(double)); err != cudaSuccess) std::cout << cudaGetErrorString(err);
	cudaMemcpy(mData, data, mN * mN * sizeof(double), cudaMemcpyHostToDevice);
}

void cudaMatrix::syncHost(double* hostData){
	cudaMemcpy(hostData, mData, mN * mN * sizeof(double), cudaMemcpyDeviceToHost);
}

void cudaMatrix::multiply(cudaMatrix &matA, cudaMatrix &matB, cudaMatrix &matC){

	constexpr unsigned int BLOCK_DIM = 32;
	constexpr dim3 blockDimension{BLOCK_DIM, BLOCK_DIM};
	dim3 gridDimension{static_cast<unsigned int>(std::ceil(static_cast<double>(matA.mN)/BLOCK_DIM)), static_cast<unsigned int>(std::ceil(static_cast<double>(matA.mN)/BLOCK_DIM))};

	if(matA.mN != matB.mN || matA.mN != matC.mN) throw std::runtime_error("Matrices are not the same size!");

	rowColProduct<<<gridDimension, blockDimension>>>(matA.mData, matB.mData, matC.mData, matA.mN);
}

cudaMatrix::~cudaMatrix(){
	cudaFree(mData);
}
