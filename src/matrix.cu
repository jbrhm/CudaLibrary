#include "matrix.hpp"

#include "cuda_matrix.cuh"

Matrix::Matrix(unsigned int N) : mCuMatrix{nullptr}, mN{N}, mMatrix{static_cast<float*>(malloc(mN * mN * sizeof(float)))} {
	// Init the Identity Matrix
	for(int row = 0; row < mN; ++row){
		for(int col = 0; col < mN; ++col){
			at(row, col) = (row == col) ? 1 : 0;
		}
	}

	mCuMatrix = new cudaMatrix(mN, mMatrix);
}

Matrix::Matrix(unsigned int N, float* data) : mCuMatrix{nullptr}, mN{N}, mMatrix{static_cast<float*>(malloc(mN * mN * sizeof(float)))} {
	// Init the Identity Matrix
	for(int row = 0; row < mN; ++row){
		for(int col = 0; col < mN; ++col){
			at(row, col) = data[row * mN + col];
		}
	}

	mCuMatrix = new cudaMatrix(mN, mMatrix);
}

unsigned int Matrix::getSize() const{
	return mN;
}

Matrix::~Matrix(){
	free(mMatrix);
}

float& Matrix::at(unsigned int row, unsigned int col){
	return mMatrix[row * mN + col];
}

float const& Matrix::at(unsigned int row, unsigned int col) const{
	return mMatrix[row * mN + col];
}

void Matrix::sync(){
	mCuMatrix->syncHost(mMatrix);
}

void Matrix::multiply(Matrix &matA, Matrix &matB, Matrix &matC){
	cudaMatrix::multiply(*matA.mCuMatrix, *matB.mCuMatrix, *matC.mCuMatrix);
}

std::ostream& operator<<(std::ostream& os, Matrix const& matrix){
	std::size_t const N = matrix.getSize();

	for(int row = 0; row < N; ++row){
		for(int col = 0; col < N; ++col){
			os << matrix.at(row, col) << ' ';
		}
		os << '\n';
	}

	return os;
}
