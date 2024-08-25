#include "immintrin.h"
#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <cmath>

class avxVector{
private:
	static constexpr float floatsPerAVX = 8.0;
	
	unsigned int mSize; // The number of "real" elements in the array
	unsigned int mAVXSize; // The number of AVX elements in the AVX array
	unsigned int mCapacity; // Since the array must store values in multiples of 8 mCapacity >= mSize

	float* mData;

	__m256* mAVXData;

public:
	avxVector(unsigned int n, float* data);

	void syncAVX(float* hostData);

	void syncHost(float* hostData);

	static void vectorAdd(avxVector* v1, avxVector* v2, avxVector* out);

	~avxVector();
};
