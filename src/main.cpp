#include "matrix.hpp"
#include <iostream>

int main(){
	unsigned int M = 1000;
	unsigned int N = 1000;
	unsigned int K = 1000;
	
	Matrix m1(M, K);

	// Matrix 2
	float* data = new float[9]{2,0,0,0,2,0,0,0,2};
	Matrix m2(K, N);

	Matrix m3(M, N);

	Matrix::mySGEMM(m1, m2, m3);

	//Matrix::cublasSGEMM(m1, m2, m3);

	Matrix::report();

	Matrix::measureFLOPS(m1, m2, m3, false);

	// Matrix::measureFLOPS(m1, m2, m3, true);

	m3.sync();

	m3.print();

	return EXIT_SUCCESS;
}
