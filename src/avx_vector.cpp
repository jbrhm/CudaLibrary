#include "avx_vector.hpp"

avxVector::avxVector(unsigned int n, float* data) : mSize{n}{
	syncAVX(data);
}

void avxVector::syncAVX(float* hostData){
	// Load the data into aligned array
	for(unsigned int i = 0; i < AVX_SIZE; ++i){
		mData[i] = (i < mSize) ? hostData[i] : 0.0;
	}

	// Load from aligned array into AVX registers
	for(unsigned int i = 0; i < AVX_SIZE; i += 8){
		mAVXData[i/8] = _mm256_load_ps(&mData[i]);
	}
}
