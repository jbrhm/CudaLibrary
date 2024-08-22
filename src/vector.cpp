#include "vector.hpp"
#include "avx_vector.hpp"

avxVector* Vector::AVXVectorFactory(unsigned int n, float* data){
	return new avxVector(n, data);
}

void Vector::syncAVX(){
	syncHost();
	if(mState != State::AVX){
		mState = State::AVX;
		mAVXVector->syncAVX(mData.data());
	}
}

void Vector::syncHostFromAVX(){
	mAVXVector->syncHost(mData.data());
}

void Vector::avxAdd(avxVector* v1, avxVector* v2, avxVector* out){
	avxVector::vectorAdd(v1, v2, out);
}
