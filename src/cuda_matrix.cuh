#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <iostream>

class cudaMatrix {
private:
	unsigned int mN;

	double* mData;

public:
	cudaMatrix(unsigned int N, double* data);

	void syncHost(double* hostData);

	static void multiply(cudaMatrix &matA, cudaMatrix &matB, cudaMatrix &matC);

	~cudaMatrix();
};
