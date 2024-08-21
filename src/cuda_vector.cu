#include "cuda_vector.cuh"

template<unsigned int READ_SIZE>
__global__ void vector_add(float* data1, float* data2, float* out, unsigned int n){
	int id = (blockIdx.x * blockDim.x + threadIdx.x) * READ_SIZE;

	unsigned int upperBound = (n <= (id + READ_SIZE)) ? n : id + READ_SIZE;

	for(unsigned int i = id; i < upperBound; ++i){
		out[i] = data1[i] + data2[i];
	}
}

cudaVector::cudaVector(unsigned int n, float* data) : mData{nullptr}, mSize{n} {
	if(cudaError_t err = cudaMalloc(&mData, sizeof(float) * mSize); err != cudaSuccess){
		throw std::runtime_error(std::format("Cuda Failed: {}", std::string(cudaGetErrorName(err))));
	}

	if(cudaError_t err = cudaMemcpy(mData, data, sizeof(float) * mSize, cudaMemcpyHostToDevice); err != cudaSuccess){
		throw std::runtime_error(std::format("Cuda Failed: {}", std::string(cudaGetErrorName(err))));
	}
}

void cudaVector::syncHost(float* hostData){
	if(cudaError_t err = cudaMemcpy(hostData, mData, sizeof(float) * mSize, cudaMemcpyDeviceToHost); err != cudaSuccess){
		throw std::runtime_error(std::format("Cuda Failed: {}", std::string(cudaGetErrorName(err))));
	}
}

void cudaVector::syncDevice(float* hostData){
	if(cudaError_t err = cudaMemcpy(hostData, mData, sizeof(float) * mSize, cudaMemcpyHostToDevice); err != cudaSuccess){
		throw std::runtime_error(std::format("Cuda Failed: {}", std::string(cudaGetErrorName(err))));
	}
}
void cudaVector::vectorAdd(cudaVector* vec1, cudaVector* vec2, cudaVector* out){
	unsigned int BLOCK_SIZE = 32;
	unsigned int GRID_SIZE = std::ceil(static_cast<float>(vec1->mSize)/(READ_SIZE * BLOCK_SIZE));

	if(vec1->mSize != vec2->mSize || vec2->mSize != out->mSize){
		throw std::runtime_error("Vector Sizes Do Not Match");
	}

	vector_add<READ_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(vec1->mData, vec2->mData, out->mData, vec1->mSize);
}

cudaVector::~cudaVector(){
	cudaFree(mData);
}

