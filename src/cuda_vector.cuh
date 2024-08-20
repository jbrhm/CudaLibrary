#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <format>
#include <string>

class cudaVector{
private:
	float* mData;
	unsigned int mSize;
		
public:
	cudaVector(unsigned int n);
};
