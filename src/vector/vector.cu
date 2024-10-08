#include "vector.hpp"
#include "cuda_vector.cuh"

Vector::Vector(unsigned int n) : mCuVector{nullptr}, mAVXVector{nullptr}, mData(n), mSize{n} /* Defaults to zero*/, mState{State::HOST} {
	mCuVector = new cudaVector(n, mData.data());
	mAVXVector = AVXVectorFactory(n, mData.data());
}

Vector::Vector(unsigned int n, float* data): mCuVector{nullptr}, mAVXVector{nullptr}, mData(n), mSize{n}, mState{State::HOST} {
	memcpy(mData.data(), data, sizeof(float) * n);
	mCuVector = new cudaVector(n, mData.data());
	mAVXVector = AVXVectorFactory(n, mData.data());
}

void Vector::syncHost(){
	if(mState == State::DEVICE){
		mState = State::HOST;
		mCuVector->syncHost(mData.data());
	}else if(mState == State::AVX){
		mState = State::AVX;
		syncHostFromAVX();
	}
}

void Vector::syncDevice(){
	if(mState != State::DEVICE){
		mState = State::DEVICE;
		mCuVector->syncHost(mData.data());
	}
}

void Vector::print(){
	
	syncHost();

	std::cout << "[ ";

	for(auto num : mData){
		std::cout << num << ' ';
	}

	std::cout << ']';
}

void Vector::print(std::ostream& os){
	
	syncHost();

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
		if(	vec1.mState == State::AVX &&
			vec2.mState == State::AVX &&
			out.mState == State::AVX
			){
			avxAdd(vec1.mAVXVector, vec2.mAVXVector, out.mAVXVector);
		}else{
			for(unsigned int i = 0; i < vec1.mSize; ++i){
				out.mData[i] = vec1.mData[i] + vec2.mData[i];
			}
		}
	}
}

Vector::~Vector(){

}
