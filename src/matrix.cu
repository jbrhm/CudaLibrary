#include "matrix.hpp"

#include "cuda_matrix.cuh"

Matrix::Matrix(unsigned int M, unsigned int N) : mState{State::HOST}, mCuMatrix{nullptr}, mM{M}, mN{N}, mMatrix{static_cast<float*>(malloc(mM * mN * sizeof(float)))} {
	// Init the Identity Matrix
	for(int row = 0; row < mM; ++row){
		for(int col = 0; col < mN; ++col){
			at(row, col) = (row == col) ? 1 : 0;
		}
	}
	
	// Create a copy of the matrix on the GPU
	mCuMatrix = new cudaMatrix(mM, mN, mMatrix);
}

Matrix::Matrix(unsigned int M, unsigned int N, float* data) : mState{State::HOST}, mCuMatrix{nullptr}, mM{M}, mN{N}, mMatrix{static_cast<float*>(malloc(mM * mN * sizeof(float)))} {
	// Init the Identity Matrix
	for(int row = 0; row < mM; ++row){
		for(int col = 0; col < mN; ++col){
			at(row, col) = data[row * mN + col];
		}
	}

	// Create a copy of the matrix on the GPU
	mCuMatrix = new cudaMatrix(mM, mN, mMatrix);
}

pair<unsigned int, unsigned int> Matrix::getSize() const{
	return pair<unsigned int, unsigned int>(mM, mN);
}

Matrix::~Matrix(){
	free(mMatrix);
}

float& Matrix::at(unsigned int row, unsigned int col){
	syncHost();

	return mMatrix[row * mN + col];
}

float const& Matrix::at(unsigned int row, unsigned int col) const{
	if(mState != State::HOST)
		throw std::runtime_error("Matrix Exists on the GPU");

	return mMatrix[row * mN + col];
}

void Matrix::syncHost(){
	if(mState != State::HOST){
		mState = State::HOST;
		mCuMatrix->syncHost(mMatrix);
	}
}

void Matrix::syncDevice(){
	if(mState != State::DEVICE){
		mState = State::DEVICE;
		mCuMatrix->syncDevice(mMatrix);
	}
}

void Matrix::mySGEMM(Matrix &matA, Matrix &matB, Matrix &matC){
	// If the matrices are large, then alloc them to the GPU
	if(	
		matA.getSize().first >= HOST_TO_CUDA_THRESHOLD ||
		matA.getSize().second >= HOST_TO_CUDA_THRESHOLD ||
		matB.getSize().first >= HOST_TO_CUDA_THRESHOLD ||
		matB.getSize().second >= HOST_TO_CUDA_THRESHOLD
	){
		matA.syncDevice();
		matB.syncDevice();
		matC.mState =State::DEVICE; // This one is set because we dont want to do any extra work to copy data from the host

		cudaMatrix::mySGEMM(*matA.mCuMatrix, *matB.mCuMatrix, *matC.mCuMatrix);
	}else{
		matA.syncHost();
		matB.syncHost();
		matC.mState =State::HOST;

		Matrix::hostSGEMM(matA, matB, matC);
	}
}

void Matrix::hostSGEMM(Matrix &matA, Matrix &matB, Matrix &matC){
	for(unsigned int row = 0; row < matA.mM; ++row){
		for(unsigned int col = 0; col < matB.mN; ++col){
			// We dont want to use the at function because that will do unecessary checking
			float sum = 0;

			for(unsigned int k = 0; k < matA.mN; ++k){
				sum += matA.mMatrix[matA.mN * row + k] * matB.mMatrix[col + matB.mN * k];
			}

			matC.mMatrix[row * matB.mN + col] = sum;
		}
	}
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
	syncHost();
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
