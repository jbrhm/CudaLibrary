#include "vector.hpp"
#include "cuda_vector.cuh"

Vector::Vector(unsigned int n) : mCuVector{nullptr}, mData(n){
	; // Defaults to zero
	mCuVector = new cudaVector(n, mData.data());
}

void Vector::syncHost(){
	mCuVector->syncHost(mData.data());
}

void Vector::syncDevice(){
	mCuVector->syncHost(mData.data());
}

void Vector::print(std::ostream& os){
	os << "[ ";

	for(auto num : mData){
		os << num << ' ';
	}

	os << ']';
}

Vector::~Vector(){

}
