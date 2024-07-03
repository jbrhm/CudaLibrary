#include "cuda_matrix.cuh"

__global__ void rowColProduct(double* dataA, double* dataB, double* dataC, unsigned int N){
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id < N * N){
		int row = id / N;
		int col = id % N;

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

	constexpr unsigned int THREADS = 32;

	if(matA.mN != matB.mN || matA.mN != matC.mN) throw std::runtime_error("Matrices are not the same size!");

	rowColProduct<<<std::ceil((matA.mN * matA.mN)/static_cast<double>(THREADS)), THREADS>>>(matA.mData, matB.mData, matC.mData, matA.mN);
}

cudaMatrix::~cudaMatrix(){
	cudaFree(mData);
}
