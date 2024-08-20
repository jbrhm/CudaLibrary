#include "vector.hpp"
#include "cuda_vector.cuh"

Vector::Vector(unsigned int n) : mCuVector{nullptr}{
	std::vector<float> arr(n); // Defaults to zero
	mCuVector = new cudaVector(n, arr.data());
}

Vector::~Vector(){

}
