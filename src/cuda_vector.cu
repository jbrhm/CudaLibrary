#include "cuda_vector.cuh"

cudaVector::cudaVector(unsigned int n) : mData{nullptr}, mSize{n} {
	if(cudaError_t err = cudaMalloc(&mData, sizeof(float) * mSize); err != cudaSuccess){
		throw std::runtime_error(std::format("Cuda Failed: {}", std::string(cudaGetErrorName(err))));
	}
}
