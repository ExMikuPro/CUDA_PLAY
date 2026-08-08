#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void add10(int* data)
{
    int i = threadIdx.x;
    data[i] = data[i] + 10;
}

int main() {
    int host_data[8] = {
        0,1,2,3,4,5,6,7
    };

    int* device_data;

    cudaMalloc(&device_data, sizeof(host_data));

    cudaMemcpy(device_data, host_data, sizeof(host_data),cudaMemcpyHostToDevice);

    add10<<<1,8>>>(device_data);

    cudaDeviceSynchronize();

    cudaMemcpy(host_data, device_data, sizeof(host_data),cudaMemcpyDeviceToHost);

    cudaFree(device_data);

    for (int value : host_data)
    {
        std::cout << value << ' ';
    }

    std::cout << '\n';

    return EXIT_SUCCESS;
}
