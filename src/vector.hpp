#pragma once
#include <vector>
#include <iostream>

class cudaVector;

class Vector{
private:
	cudaVector* mCuVector;

	std::vector<float> mData;
public:
	Vector(unsigned int n);

	void syncHost();

	void syncDevice();

	void print(std::ostream& os);

	~Vector();
};
