#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void add10(int* data, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        data[i] = data[i] + 10;
    }
}

int main() {
    constexpr int size = 32;
    int host_data[size];
    for (int i = 0; i < size; ++i)
    {
        host_data[i] = i;
    }

    int* device_data;

    cudaMalloc(&device_data, sizeof(host_data));

    cudaMemcpy(device_data, host_data, sizeof(host_data), cudaMemcpyHostToDevice);

    constexpr int block_size = 8;

    int block_count = (size + block_size - 1) / block_size;

    add10<<<block_count, block_size>>>(device_data, size);

    cudaDeviceSynchronize();

    cudaMemcpy(host_data, device_data, sizeof(host_data), cudaMemcpyDeviceToHost);

    cudaFree(device_data);

    for (int value : host_data)
    {
        std::cout << value << " ";
    }

    std::cout << '\n';

    return EXIT_SUCCESS;
}
