#include "cuda_vector.cuh"

cudaVector::cudaVector(unsigned int n, float* data) : mData{nullptr}, mSize{n} {
	if(cudaError_t err = cudaMalloc(&mData, sizeof(float) * mSize); err != cudaSuccess){
		throw std::runtime_error(std::format("Cuda Failed: {}", std::string(cudaGetErrorName(err))));
	}

	if(cudaError_t err = cudaMemcpy(mData, data, sizeof(float) * mSize, cudaMemcpyHostToDevice); err != cudaSuccess){
		throw std::runtime_error(std::format("Cuda Failed: {}", std::string(cudaGetErrorName(err))));
	}
}

cudaVector::~cudaVector(){
	cudaFree(mData);
}
