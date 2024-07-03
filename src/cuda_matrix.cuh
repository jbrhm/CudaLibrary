#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <iostream>

class cudaMatrix {
private:
	unsigned int mN;

	float* mData;

public:
	cudaMatrix(unsigned int N, float* data);

	void syncHost(float* hostData);

	static void multiply(cudaMatrix &matA, cudaMatrix &matB, cudaMatrix &matC);

	~cudaMatrix();
};
