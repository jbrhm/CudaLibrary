#include "matrix.hpp"

extern "C" {
    unsigned long long new_matrix(int M, int N){
		return reinterpret_cast<unsigned long long>(new Matrix(M, N));
	}

    unsigned long long new_matrix_from_data(int M, int N, float* data){
		return reinterpret_cast<unsigned long long>(new Matrix(M, N, data));
	}

	void sync(unsigned long long matrix){
		Matrix* matrixP = reinterpret_cast<Matrix*>(matrix);
		matrixP->syncHost();
	}

	void print(unsigned long long matrix){
		Matrix* matrixP = reinterpret_cast<Matrix*>(matrix);
		matrixP->print();
	}

	void multiply(unsigned long long A, unsigned long long B, unsigned long long C){
		
		Matrix* matrixA = reinterpret_cast<Matrix*>(A);
		Matrix* matrixB = reinterpret_cast<Matrix*>(B);
		Matrix* matrixC = reinterpret_cast<Matrix*>(C);

		Matrix::mySGEMM(*matrixA, *matrixB, *matrixC);

	}

	void release(unsigned long long matrix){
		delete reinterpret_cast<Matrix*>(matrix);
	}
}
