#include "matrix.hpp"

extern "C" {
    Matrix* new_matrix(int M, int N){
		std::cout << "We made it" << std::endl;
		return new Matrix(M, N);
	}

    Matrix* new_matrix_from_data(int M, int N, float* data){
		return new Matrix(M, N, data);
	}

	void sync(Matrix* matrix){
		matrix->sync();
	}

	void print(Matrix* matrix){
		matrix->print();
	}
}
