#pragma once
#include <cstdint>
#include <cstdlib>
#include <ostream>

class cudaMatrix;


class Matrix {
private:
	// Pointer to gpu matrix
	cudaMatrix* mCuMatrix;

	unsigned int mN;

	double* mMatrix;

public:

	Matrix(unsigned int N);

	Matrix(unsigned int N, double* data);

	unsigned int getSize() const;

	double& at(unsigned int row, unsigned int col);

	double const& at(unsigned int row, unsigned int col) const;

	void sync();

	static void multiply(Matrix &matA, Matrix &matB, Matrix &matC);

	~Matrix();
};

std::ostream& operator<<(std::ostream& os, Matrix const& matrix);
