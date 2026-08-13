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
    const int size = 1000003;

    std::vector<int> host_a(size);
    std::vector<int> host_b(size);
    std::vector<int> host_c(size, 0);

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

    CUDA_CHECK(cudaMalloc(&device_b, bytes));

    CUDA_CHECK(cudaMalloc(&device_c, bytes));

    CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(device_c, host_c.data(), bytes, cudaMemcpyHostToDevice));

    constexpr int block_size = 256;

    int block_count = (size + block_size - 1) / block_size;

    vectorAdd<<<block_count,block_size>>>(
        device_a, device_b, device_c, size
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(host_a.data(), device_a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_b.data(), device_b, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; ++i)
    {
        std::cout << host_a[i] << " + " << host_b[i] << " = " << host_c[i] << std::endl;
    }

    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_c));

    return EXIT_SUCCESS;
}
