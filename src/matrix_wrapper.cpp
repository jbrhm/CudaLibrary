#include "matrix.hpp"

extern "C" {
    unsigned long long new_matrix(int M, int N){
		std::cout << "We made it" << std::endl;
		return reinterpret_cast<unsigned long long>(new Matrix(M, N));
	}

    Matrix* new_matrix_from_data(int M, int N, float* data){
		return new Matrix(M, N, data);
	}

	void sync(unsigned long long matrix){
		Matrix* matrixP = reinterpret_cast<Matrix*>(matrix);
		matrixP->sync();
	}

	void print(unsigned long long matrix){
		std::cout << "Pre print l " << matrix << "\n";
		Matrix* matrixP = reinterpret_cast<Matrix*>(matrix);
		std::cout << "Pre print " << matrixP << "\n";
		matrixP->print();
	}
}
