#include "vector.hpp"
#include "cuda_vector.cuh"

Vector::Vector(unsigned int n) : mCuVector{nullptr}, mData(n), mSize{n} /* Defaults to zero*/, mState{State::HOST} {
	mCuVector = new cudaVector(n, mData.data());
}

Vector::Vector(unsigned int n, float* data): mCuVector{nullptr}, mData(n), mSize{n}, mState{State::HOST} {
	memcpy(mData.data(), data, sizeof(float) * n);
	mCuVector = new cudaVector(n, mData.data());
}

void Vector::syncHost(){
	if(mState != State::HOST){
		mCuVector->syncHost(mData.data());
	}
}

void Vector::syncDevice(){
	if(mState != State::DEVICE){
		mCuVector->syncHost(mData.data());
	}
}

void Vector::print(std::ostream& os){
	os << "[ ";

	for(auto num : mData){
		os << num << ' ';
	}

	os << ']';
}

void Vector::vectorAdd(Vector& vec1, Vector& vec2, Vector& out){
	if(	vec1.mSize >= HOST_TO_CUDA_THRESHOLD ||
		vec2.mSize >= HOST_TO_CUDA_THRESHOLD ||
		out.mSize >= HOST_TO_CUDA_THRESHOLD
			){
		
		// Sync to the GPU
		vec1.syncDevice();
		vec2.syncDevice();
		out.syncDevice();

		cudaVector::vectorAdd(vec1.mCuVector, vec2.mCuVector, out.mCuVector);
	}else{
		//TODO: Implement
		exit(EXIT_FAILURE);
	}
}

Vector::~Vector(){

}
