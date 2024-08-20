#pragma once
#include <vector>

class cudaVector;

class Vector{
private:
	cudaVector* mCuVector;
public:
	Vector(unsigned int n);

	void syncHost();

	void syncDevice();

	~Vector();
};
