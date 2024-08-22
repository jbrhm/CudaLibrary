#include "immintrin.h"

class avxVector{
private:
	unsigned int mSize;

	// AVX
	constexpr static unsigned int AVX_SIZE = 256;
	alignas(32) float mData[256];

	__m256 mAVXData[32];

public:
	avxVector(unsigned int n, float* data);

	void syncAVX(float* hostData);

	~avxVector();
};
