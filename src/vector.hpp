#pragma once
#include <vector>
#include <iostream>
#include <algorithm>

class cudaVector;
class avxVector;

class Vector{
private:	
	enum State {
		HOST = 0,
		DEVICE = 1,
		AVX = 2
	}; 

	cudaVector* mCuVector;
	alignas(32) avxVector* mAVXVector;

	std::vector<float> mData;

	unsigned int mSize;

	State mState;

	//TODO: Tune this threshold
	constexpr static unsigned int HOST_TO_CUDA_THRESHOLD = 256;


public:
	Vector(unsigned int n);

	Vector(unsigned int n, float* data);

	void syncHost();

	void syncDevice();

	// AVX
	
	avxVector* AVXVectorFactory(unsigned int n, float* data);
	
	void syncAVX();

	void syncHostFromAVX();

	static void avxAdd(avxVector* v1, avxVector* v2, avxVector* out);

	// AVX

	void print(std::ostream& os);

	static void vectorAdd(Vector& vec1, Vector& vec2, Vector& out);

	~Vector();
};
