#include "cuda_matrix.cuh"

LoopProfiler cudaMatrix::mLoopProfiler{};

template<unsigned int TILE_DIM>
__global__ void rowColProduct(float* dataA, float* dataB, float* dataC, unsigned int M, unsigned int N, unsigned int K){

	// The final location in dataC that is being computed
	// These coordinates are in tile space not matrix space
	unsigned int const row = blockIdx.x;
	unsigned int const col = blockIdx.y;

	// Allocate shared buffers
	__shared__ float sharedA[TILE_DIM * TILE_DIM];
	__shared__ float sharedB[TILE_DIM * TILE_DIM];

	// Use the global memory coalescing pattern to get the thread id
	unsigned int const threadCol = threadIdx.x % TILE_DIM;
	unsigned int const threadRow = threadIdx.x / TILE_DIM;

	// These are the top right corner of all of the matrices
	dataA += row * TILE_DIM * K;
	dataB += col * TILE_DIM;
	dataC += row * TILE_DIM * N + col * TILE_DIM;

	float dot = 0;
	
	// Iterate through all of the blocks requried to process the entire matrix
	for(unsigned int blockNum = 0; blockNum < K; blockNum += TILE_DIM){
		
		// Populate the shared memory
		sharedA[threadRow * TILE_DIM + threadCol] = dataA[threadRow * K + threadCol];
		sharedB[threadRow * TILE_DIM + threadCol] = dataB[threadRow * N + threadCol];

		__syncthreads();

		// Move to the next chunk
		dataA += TILE_DIM;
		dataB += TILE_DIM * N;

		// Dot for current SMEM
		for(unsigned int i = 0; i < TILE_DIM; ++i){
			dot += sharedA[threadRow * TILE_DIM + i] * sharedB[i * TILE_DIM + threadCol];
		}

		__syncthreads();
	}

	dataC[threadRow * N + threadCol] = dot;
}

cudaMatrix::cudaMatrix(unsigned int M, unsigned int N, float* data) : mN{N}, mM{M}, mData{nullptr} {
	if(cudaError_t err = cudaMalloc(&mData, mN * mM * sizeof(float)); err != cudaSuccess) std::cout << cudaGetErrorString(err);
	cudaMemcpy(mData, data, mN * mM * sizeof(float), cudaMemcpyHostToDevice);
}

void cudaMatrix::syncHost(float* hostData){
	cudaMemcpy(hostData, mData, mN * mM * sizeof(float), cudaMemcpyDeviceToHost);
}

void cudaMatrix::syncDevice(float* hostData){
	cudaMemcpy(hostData, mData, mN * mM * sizeof(float), cudaMemcpyDeviceToHost);
}

void cudaMatrix::mySGEMM(cudaMatrix &matA, cudaMatrix &matB, cudaMatrix &matC){

	constexpr unsigned int BLOCK_DIM = 32;
	constexpr dim3 blockDimension{BLOCK_DIM * BLOCK_DIM};
	dim3 gridDimension{static_cast<unsigned int>(std::ceil(static_cast<float>(matA.mM)/BLOCK_DIM)), static_cast<unsigned int>(std::ceil(static_cast<float>(matB.mN)/BLOCK_DIM))};

	if(matA.mN != matB.mM) throw std::runtime_error("Matrices are not compatable!");

	mLoopProfiler.start("my");

	rowColProduct<BLOCK_DIM><<<gridDimension, blockDimension>>>(matA.mData, matB.mData, matC.mData, matA.mM, matB.mN, matA.mN);

	mLoopProfiler.finish("my");
}

void cudaMatrix::cublasSGEMM(cudaMatrix &matA, cudaMatrix &matB, cudaMatrix &matC){
	cublasHandle_t handle;
	cublasCreate(&handle);

	const float alpha = 1.0f;
	const float beta = 0.0f;

	mLoopProfiler.start("cuBLAS");

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matA.mN, matA.mN, matA.mN, &alpha, matA.mData, matA.mN, matB.mData, matA.mN, &beta, matC.mData, matA.mN);

	mLoopProfiler.finish("cuBLAS");
}

void cudaMatrix::report(std::string const& name){
	std::cout << " has had an average performance of " << name << " " << mLoopProfiler.avg(name) << " seconds\n";
}

void cudaMatrix::measureFLOPS(cudaMatrix &matA, cudaMatrix &matB, cudaMatrix &matC, bool isCuBLAS){
	size_t flops = 2 * matA.mM * matA.mN * matB.mN;

	constexpr unsigned int iterations = 1000;

	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

	if(isCuBLAS){
		for (size_t i = 0; i < iterations; ++i) {
			cudaMatrix::cublasSGEMM(matA, matB, matC);
		}
	}else{
		for (size_t i = 0; i < iterations; ++i) {
			cudaMatrix::mySGEMM(matA, matB, matC);
		}
	}

	std::chrono::duration<double> elapsedSeconds = std::chrono::system_clock::now() - start;

	double GFLOPS = (iterations * flops * 1e-9) / elapsedSeconds.count();
	if(isCuBLAS){
		std::cout << "cuBLAS " << GFLOPS << " GFLOPS\n";
	}else{
		std::cout << "my " << GFLOPS << " GFLOPS\n";
	}
}

cudaMatrix::~cudaMatrix(){
	cudaFree(mData);
}
