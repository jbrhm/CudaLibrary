#pragma once
#include "matrix.hpp"
#include <iostream>

int main(){
	unsigned int size = 1000;
	Matrix m1(size);

	// Matrix 2
	float* data = new float[9]{2,0,0,0,2,0,0,0,2};
	Matrix m2(size);

	Matrix m3(size);

	Matrix::multiply(m1, m2, m3);

	m3.sync();

	std::cout << m3;

	return EXIT_SUCCESS;
}
