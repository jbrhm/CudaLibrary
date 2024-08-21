#pragma once
#include <vector>
#include <iostream>
#include <algorithm>

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
	constexpr static unsigned int HOST_TO_CUDA_THRESHOLD = 250;
public:
	Vector(unsigned int n);

	Vector(unsigned int n, float* data);

	void syncHost();

	void syncDevice();

	void print(std::ostream& os);

	static void vectorAdd(Vector& vec1, Vector& vec2, Vector& out);

	~Vector();
};
