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

	float* mMatrix;

public:

	Matrix(unsigned int N);

	Matrix(unsigned int N, float* data);

	unsigned int getSize() const;

	float& at(unsigned int row, unsigned int col);

	float const& at(unsigned int row, unsigned int col) const;

	void sync();

	static void multiply(Matrix &matA, Matrix &matB, Matrix &matC);

	~Matrix();
};

std::ostream& operator<<(std::ostream& os, Matrix const& matrix);
