#include "scan.hpp"

#define CUPYBARA_SCAN_BLOCK_SIZE 256

__global__ void device_scan(int* input, int* output, unsigned long long len) {

    // thread x
    const int tx = threadIdx.x;
    int id = blockIdx.x * blockDim.x + tx;

    // allocate buffer in shared memory
    __shared__ int scan_array[CUPYBARA_SCAN_BLOCK_SIZE];

    if(id < len){
        // copy the array into shared memory
        scan_array[tx] = input[id];

        int stride = 1;
        while(stride < CUPYBARA_SCAN_BLOCK_SIZE){
            const int index = (tx + 1) * stride * 2 - 1;

            // have it so that the final thread operating on the last element of the array is the one which is active
            if(index < CUPYBARA_SCAN_BLOCK_SIZE){
                scan_array[index] += scan_array[index - stride];
            }

            // double the stride
            stride *= 2;

            // sync between reduction steps
            __syncthreads();
        }

        stride = CUPYBARA_SCAN_BLOCK_SIZE >> 2;

        while(stride > 0){
            const int index = (tx + 1) * stride * 2 - 1;

            if(index + stride < CUPYBARA_SCAN_BLOCK_SIZE){
                scan_array[index + stride] += scan_array[index];
            }

            stride = stride >> 1;

            __syncthreads();
        }

            
        // copy the array into shared memory
        output[id] = scan_array[tx];
    }
}

scan::scan(int* _data, std::size_t _len, bool _is_inclusive) : data{_data}, len{_len}, is_inclusive{_is_inclusive}, device_input{nullptr}, device_output{nullptr}, output{}{

    // TODO: add checks to make sure these operations succeed
    // allocate on the gpu
    cudaMalloc(reinterpret_cast<void**>(&device_input), sizeof(int) * len);
    cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(int) * len);

    // copy the data to be scanned to the GPU
    cudaMemcpy(device_input, data, sizeof(int) * len, cudaMemcpyHostToDevice);

    dim3 blocks(CUPYBARA_SCAN_BLOCK_SIZE);
    dim3 grids(std::ceil(static_cast<float>(len) / CUPYBARA_SCAN_BLOCK_SIZE));

    // enqueue the scan
    device_scan<<<grids, blocks>>>(device_input, device_output, len);

    // sync the device
    cudaDeviceSynchronize();

    // copy the data back
    cudaMemcpy(output.data(), device_output, sizeof(int) * len, cudaMemcpyDeviceToHost);
}

void scan::print(){
    for(auto const& d : output){
        // print each thing in the scan
        std::cout << d << ' ';
    }

    // print new line
    std::cout << '\n';
}
