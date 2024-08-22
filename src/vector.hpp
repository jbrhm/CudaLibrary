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
	avxVector* mAVXVector;

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

	void syncAVX();

	void print(std::ostream& os);

	static void vectorAdd(Vector& vec1, Vector& vec2, Vector& out);

	~Vector();
};
