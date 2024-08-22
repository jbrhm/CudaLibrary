#include "vector.hpp"

extern "C" {
    unsigned long long new_vector(int n){
		return reinterpret_cast<unsigned long long>(new Vector(n));
	}

    unsigned long long new_vector_from_data(int n, float* data){
		return reinterpret_cast<unsigned long long>(new Vector(n, data));
	}

	void vector_syncHost(unsigned long long vector){
		Vector* vectorPtr = reinterpret_cast<Vector*>(vector);
		vectorPtr->syncHost();
	}

	void vector_syncAVX(unsigned long long vector){
		Vector* vectorPtr = reinterpret_cast<Vector*>(vector);
		vectorPtr->syncAVX();
	}

	void vector_print(unsigned long long vector){
		Vector* vectorPtr = reinterpret_cast<Vector*>(vector);
		vectorPtr->print();
	}

	void vector_add(unsigned long long A, unsigned long long B, unsigned long long C){
		Vector* matrixA = reinterpret_cast<Vector*>(A);
		Vector* matrixB = reinterpret_cast<Vector*>(B);
		Vector* matrixC = reinterpret_cast<Vector*>(C);

		Vector::vectorAdd(*matrixA, *matrixB, *matrixC);
	}

	void vector_release(unsigned long long vector){
		delete reinterpret_cast<Vector*>(vector);
	}
}
