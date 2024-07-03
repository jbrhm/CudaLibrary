#pragma once
#include "matrix.hpp"
#include <iostream>

int main(){
	Matrix m1(3);

	// Matrix 2
	double* data = new double[9]{2,0,0,0,2,0,0,0,2};
	Matrix m2(3, data);

	Matrix m3(3);

	Matrix::multiply(m1, m2, m3);

	m3.sync();

	std::cout << m3;

	return EXIT_SUCCESS;
}
