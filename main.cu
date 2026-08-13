#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
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

__global__ void renderMandelbrot(unsigned char* pixels, int width, int height, int max_iterations)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    double cx = -2.5 + (static_cast<double>(x) / static_cast<double>(width)) * 3.5;

    double cy = -1.0 + (static_cast<double>(y) / static_cast<double>(height)) * 2.0;

    double zx = 0.0;
    double zy = 0.0;

    int iteration = 0;

    while (zx * zx + zy * zy <= 4.0 && iteration < max_iterations)
    {
        double new_zx = zx * zx - zy * zy + cx;

        double new_zy = 2.0 * zx * zy + cy;

        zx = new_zx;
        zy = new_zy;

        ++iteration;
    }
    unsigned char r;
    unsigned char g;
    unsigned char b;

    if (iteration == max_iterations)
    {
        r = 0;
        g = 0;
        b = 0;
    }
    else
    {
        unsigned char value = static_cast<unsigned char>(
            255.0 * iteration / max_iterations
        );

        r = value;
        g = value;
        b = value;
    }

    int index = (y * width + x) * 3;

    pixels[index + 0] = r;
    pixels[index + 1] = g;
    pixels[index + 2] = b;
}

int main()
{
    const int width = 1920;

    const int height = 1080;

    const int max_iterations = 200;

    const std::size_t image_size = width * height * 3 * sizeof(unsigned char);

    const std::size_t image_bytes = width * height * 3;

    std::vector<unsigned char> host_pixels(
        width * height * 3);

    unsigned char* device_pixels = nullptr;

    CUDA_CHECK(cudaMalloc(&device_pixels, image_size));

    dim3 block_size(16, 16);

    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);


    std::cout << "Grid:" << grid_size.x << "x" << grid_size.y << std::endl;

    std::cout << "Block:" << block_size.x << "x" << block_size.y << std::endl;

    renderMandelbrot<<<grid_size, block_size>>>(device_pixels, width, height, max_iterations);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        host_pixels.data(),
        device_pixels,
        image_size,
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(device_pixels));

    std::ofstream file(
        "mandlbrot.ppm",
        std::ios::binary
    );

    if (!file)
    {
        std::cerr << "无法创建 ppm 文件" << std::endl;
        return EXIT_FAILURE;
    }

    file
        << "P6\n"
        << width
        << " "
        << height
        << "\n255\n";

    file.write(
        reinterpret_cast<const char*>(
            host_pixels.data()
        ),
        static_cast<std::streamsize>(
            image_bytes
        )
    );

    file.close();

    std::cout << "图片保存为 mandelbrot.ppm" << std::endl;


    return EXIT_SUCCESS;
}
