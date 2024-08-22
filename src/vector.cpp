#include "vector.hpp"
#include "avx_vector.hpp"

void Vector::syncAVX(){
	syncHost();
	if(mState != State::AVX){
		mState = State::AVX;
	}
}
