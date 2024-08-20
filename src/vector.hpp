#pragma once
#include <vector>
#include <iostream>
#include <algorithm>

class cudaVector;

class Vector{
private:
	cudaVector* mCuVector;

	std::vector<float> mData;
public:
	Vector(unsigned int n);

	Vector(unsigned int n, float* data);

	void syncHost();

	void syncDevice();

	void print(std::ostream& os);

	~Vector();
};
