#pragma once
#include <stdio.h>

class cudaMatrix;

// This cannot be std::pair because c doesnt know about it
template<typename FIRST_TYPE, typename SECOND_TYPE>
struct pair {
	FIRST_TYPE first;
	SECOND_TYPE second;
	pair(FIRST_TYPE _first, SECOND_TYPE _second) : first{_first}, second{_second} {}
};

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

	pair<unsigned int, unsigned int> getSize() const;

	float& at(unsigned int row, unsigned int col);

	float const& at(unsigned int row, unsigned int col) const;

	void sync();

	static void mySGEMM(Matrix &matA, Matrix &matB, Matrix &matC);

	static void cublasSGEMM(Matrix &matA, Matrix &matB, Matrix &matC);

	static void report();

	static void measureFLOPS(Matrix &matA, Matrix &matB, Matrix &matC, bool isCuBLAS);

	void print();

	~Matrix();
};