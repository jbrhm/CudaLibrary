#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <string>
#include <chrono>

#include "loop_profiler.cuh"

class cudaMatrix {
private:
	unsigned int mM;

	unsigned int mN;

	float* mData;

	static LoopProfiler mLoopProfiler;

public:
	cudaMatrix(unsigned int M, unsigned int N, float* data);

	void syncHost(float* hostData);
	
	void syncDevice(float* hostData);

	static void mySGEMM(cudaMatrix &matA, cudaMatrix &matB, cudaMatrix &matC);

	static void cublasSGEMM(cudaMatrix &matA, cudaMatrix &matB, cudaMatrix &matC);

	static void report(std::string const& name);

	static void measureFLOPS(cudaMatrix &matA, cudaMatrix &matB, cudaMatrix &matC, bool isCuBLAS);

	~cudaMatrix();
};
