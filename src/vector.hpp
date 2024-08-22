#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include "immintrin.h"

class cudaVector;

class Vector{
private:	
	enum State {
		HOST = 0,
		DEVICE = 1,
		AVX = 2
	}; 

	cudaVector* mCuVector;

	std::vector<float> mData;

	unsigned int mSize;

	State mState;

	//TODO: Tune this threshold
	constexpr static unsigned int HOST_TO_CUDA_THRESHOLD = 256;

	// AVX
	constexpr static unsigned int AVX_SIZE = 256;
	alignas(32) float mFloatData[256];

	__m256 mAVXData[32];

public:
	Vector(unsigned int n);

	Vector(unsigned int n, float* data);

	void syncHost();

	void syncDevice();

	void syncAVX();

	void print(std::ostream& os);

	static void vectorAdd(Vector& vec1, Vector& vec2, Vector& out);

	~Vector();
};
