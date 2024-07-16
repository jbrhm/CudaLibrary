#include "matrix.hpp"
#include "test_suite.hpp"
#include <iostream>

int main(){
	// Run the test suite
	run();
	
	unsigned int M = 3;
	unsigned int N = 1000;
	unsigned int K = 3;
	
	Matrix m1(M, K);

	// Matrix 2
	float* data = new float[9]{2,0,0,0,2,0,0,0,2};
	Matrix m2(K, N);
	m2.at(0,0) = 3;
	m2.at(1,1) = 3;
	m2.at(2,2) = 3;

	Matrix m3(M, N);

	Matrix::mySGEMM(m1, m2, m3);

	//Matrix::cublasSGEMM(m1, m2, m3);

	Matrix::report();

	Matrix::measureFLOPS(m1, m2, m3, false);

	// Matrix::measureFLOPS(m1, m2, m3, true);

	m3.print();

	return EXIT_SUCCESS;
}
