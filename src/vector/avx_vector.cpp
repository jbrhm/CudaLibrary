#include "avx_vector.hpp"

// aligned_alloc(32, sizeof(float) * AVX_SIZE)) is used because this itself is aligned
avxVector::avxVector(unsigned int n, float* data) : mSize{n}, AVX_SIZE{(mSize % 8 == 0) ? mSize : (mSize + (8 - (mSize % 8)))}, NUM_AVX{AVX_SIZE/8}, mData{reinterpret_cast<float*>(aligned_alloc(32, sizeof(float) * AVX_SIZE))}, mAVXData{reinterpret_cast<__m256*>(aligned_alloc(32, sizeof(__m256) * NUM_AVX))}{
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

void avxVector::syncHost(float* hostData){
	// Load from AVX registers into aligned array
	for(unsigned int i = 0; i < AVX_SIZE; i += 8){
		_mm256_store_ps(&mData[i], mAVXData[i/8]);
	}

	// Load the data onto host
	memcpy(hostData, mData, sizeof(float) * mSize);
}

void avxVector::vectorAdd(avxVector* v1, avxVector* v2, avxVector* out){
	for(unsigned int i = 0; i < v1->NUM_AVX; ++i){
		out->mAVXData[i] = _mm256_add_ps(v1->mAVXData[i], v2->mAVXData[i]);
	}
}
