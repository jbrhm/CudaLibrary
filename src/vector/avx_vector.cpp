#include "avx_vector.hpp"

// aligned_alloc(32, sizeof(float) * mCapacity)) is used because this class itself is unaligned
avxVector::avxVector(unsigned int n, float* data) : mSize{n}, mAVXSize{static_cast<unsigned int>(std::ceil(n/floatsPerAVX))}, mCapacity{static_cast<unsigned int>(std::ceil(n/floatsPerAVX) * floatsPerAVX)}, mData{reinterpret_cast<float*>(aligned_alloc(32, sizeof(float) * mSize))}, mAVXData{reinterpret_cast<__m256*>(aligned_alloc(32, sizeof(float) * mCapacity))}{
	syncAVX(data);
}

void avxVector::syncAVX(float* hostData){
	// Load the data into aligned array
	for(unsigned int i = 0; i < mCapacity; ++i){
		mData[i] = (i < mSize) ? hostData[i] : 0.0;
	}

	// Load from aligned array into AVX registers
	for(unsigned int i = 0; i < mCapacity; i += 8){
		mAVXData[i/8] = _mm256_load_ps(&mData[i]);
	}
}

void avxVector::syncHost(float* hostData){
	// Load from AVX registers into aligned array
	for(unsigned int i = 0; i < mCapacity; i += 8){
		_mm256_store_ps(&mData[i], mAVXData[i/8]);
	}

	// Load the data onto host
	memcpy(hostData, mData, sizeof(float) * mSize);
}

void avxVector::vectorAdd(avxVector* v1, avxVector* v2, avxVector* out){
	for(unsigned int i = 0; i < v1->mAVXSize; ++i){
		out->mAVXData[i] = _mm256_add_ps(v1->mAVXData[i], v2->mAVXData[i]);
	}
}
