#include "immintrin.h"
#include <cstring>
#include <iostream>
#include <stdlib.h>

class avxVector{
private:
	unsigned int mSize;

	// AVX
	constexpr static unsigned int AVX_SIZE = 256;
	constexpr static unsigned int NUM_AVX = 32;

	float* mData;

	__m256* mAVXData;

public:
	avxVector(unsigned int n, float* data);

	void syncAVX(float* hostData);

	void syncHost(float* hostData);

	static void vectorAdd(avxVector* v1, avxVector* v2, avxVector* out);

	~avxVector();
};
