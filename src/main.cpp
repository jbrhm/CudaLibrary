#pragma once
#include "matrix.hpp"
#include <iostream>

int main(){
	unsigned int size = 40;
	Matrix m1(size);

	// Matrix 2
	float* data = new float[9]{2,0,0,0,2,0,0,0,2};
	Matrix m2(size);

	Matrix m3(size);

	Matrix::mySGEMM(m1, m2, m3);

	Matrix::cublasSGEMM(m1, m2, m3);

	Matrix::report();

	Matrix::measureFLOPS(m1, m2, m3, false);

	Matrix::measureFLOPS(m1, m2, m3, true);

	m3.sync();

	return EXIT_SUCCESS;
}
