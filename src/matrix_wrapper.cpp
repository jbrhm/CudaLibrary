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
		matrixP->sync();
	}

	void print(unsigned long long matrix){
		Matrix* matrixP = reinterpret_cast<Matrix*>(matrix);
		matrixP->print();
	}

	unsigned long long multiply(unsigned long long A, unsigned long long B){
		
		Matrix* matrixA = reinterpret_cast<Matrix*>(A);
		Matrix* matrixB = reinterpret_cast<Matrix*>(B);

		Matrix* matrixC = new Matrix(matrixA->getSize().first, matrixB->getSize().second);

		Matrix::mySGEMM(*matrixA, *matrixB, *matrixC);

		return reinterpret_cast<unsigned long long>(matrixC);
	}

	void release(unsigned long long matrix){
		delete reinterpret_cast<Matrix*>(matrix);
	}
}
