#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <format>
#include <string>
#include <cmath>

constexpr static unsigned int READ_SIZE = 100;

class cudaVector{
private:
	float* mData;
	unsigned int mSize;
		
public:
	cudaVector(unsigned int n, float* data);

	void syncHost(float* hostData);

	void syncDevice(float* hostData);

	static void vectorAdd(cudaVector* vec1, cudaVector* vec2, cudaVector* out);
	
	~cudaVector();
};
