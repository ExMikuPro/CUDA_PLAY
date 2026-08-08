#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)\
    do \
    {\
    cudaError_t error = (call); \
    if (error != cudaSuccess){ \
    std::cerr << "CUDA 错误:" << cudaGetErrorString(error)<<"\n"; \
    std::exit(EXIT_FAILURE);\
    }\
    }while (0)

__global__ void vectorAdd(const int* a, const int* b, int* c, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int size = 32;

    int host_a[size], host_b[size];
    int host_c[size] = {0};

    for (int i = 0; i < size; ++i)
    {
        host_a[i] = i;

        host_b[i] = i * 10;
    }

    int* device_a = nullptr;
    int* device_b = nullptr;
    int* device_c = nullptr;

    const std::size_t bytes = size * sizeof(int);

    CUDA_CHECK(cudaMalloc(&device_a, bytes));

    cudaMalloc(&device_b, bytes);

    cudaMalloc(&device_c, bytes);

    cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(device_c, host_c, bytes, cudaMemcpyHostToDevice);

    constexpr int block_size = 8;

    int block_count = (size + block_size - 1) / block_size;

    vectorAdd<<<block_count,block_size>>>(
        device_a, device_b, device_c, size
    );

    cudaDeviceSynchronize();

    cudaMemcpy(host_a, device_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b, device_b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i)
    {
        std::cout << host_a[i] << " + " << host_b[i] << " = " << host_c[i] << std::endl;
    }

    cudaFree(host_a);
    cudaFree(host_b);
    cudaFree(host_c);

    return EXIT_SUCCESS;
}
