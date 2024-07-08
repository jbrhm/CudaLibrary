#include "matrix.hpp"

#include "cuda_matrix.cuh"

Matrix::Matrix(unsigned int M, unsigned int N) : mCuMatrix{nullptr}, mM{M}, mN{N}, mMatrix{static_cast<float*>(malloc(mM * mN * sizeof(float)))} {
	// Init the Identity Matrix
	for(int row = 0; row < mM; ++row){
		for(int col = 0; col < mN; ++col){
			at(row, col) = (row == col) ? 1 : 0;
		}
	}

	mCuMatrix = new cudaMatrix(mM, mN, mMatrix);
}

Matrix::Matrix(unsigned int M, unsigned int N, float* data) : mCuMatrix{nullptr}, mM{M}, mN{N}, mMatrix{static_cast<float*>(malloc(mM * mN * sizeof(float)))} {
	// Init the Identity Matrix
	for(int row = 0; row < mM; ++row){
		for(int col = 0; col < mN; ++col){
			at(row, col) = data[row * mN + col];
		}
	}

	mCuMatrix = new cudaMatrix(mM, mN, mMatrix);
}

pair<unsigned int, unsigned int> Matrix::getSize() const{
	return pair<unsigned int, unsigned int>(mM, mN);
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

void Matrix::mySGEMM(Matrix &matA, Matrix &matB, Matrix &matC){
	cudaMatrix::mySGEMM(*matA.mCuMatrix, *matB.mCuMatrix, *matC.mCuMatrix);
}

void Matrix::cublasSGEMM(Matrix &matA, Matrix &matB, Matrix &matC){
	cudaMatrix::cublasSGEMM(*matA.mCuMatrix, *matB.mCuMatrix, *matC.mCuMatrix);
}

void Matrix::report(){
	cudaMatrix::report("my");
	//cudaMatrix::report("cuBLAS");
}

void Matrix::measureFLOPS(Matrix &matA, Matrix &matB, Matrix &matC, bool isCuBLAS){
	cudaMatrix::measureFLOPS(*matA.mCuMatrix, *matB.mCuMatrix, *matC.mCuMatrix, isCuBLAS);
}

void Matrix::print() {
	auto const& size = getSize();
	unsigned int const N = size.second;
	unsigned int const M = size.first;

	for(int row = 0; row < M; ++row){
		for(int col = 0; col < N; ++col){
			std::cout << at(row, col) << ' ';
		}
		std::cout << '\n';
	}

}
