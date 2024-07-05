#pragma once
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <utility>

class cudaMatrix;


class Matrix {
private:
	// Pointer to gpu matrix
	cudaMatrix* mCuMatrix;

	unsigned int mM;

	unsigned int mN;

	float* mMatrix;

public:

	Matrix(unsigned int M, unsigned int N);

	Matrix(unsigned int M, unsigned int N, float* data);

	std::pair<unsigned int, unsigned int> getSize() const;

	float& at(unsigned int row, unsigned int col);

	float const& at(unsigned int row, unsigned int col) const;

	void sync();

	static void mySGEMM(Matrix &matA, Matrix &matB, Matrix &matC);

	static void cublasSGEMM(Matrix &matA, Matrix &matB, Matrix &matC);

	static void report();

	static void measureFLOPS(Matrix &matA, Matrix &matB, Matrix &matC, bool isCuBLAS);

	~Matrix();
};

std::ostream& operator<<(std::ostream& os, Matrix const& matrix);
