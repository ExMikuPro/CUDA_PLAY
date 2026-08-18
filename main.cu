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
        // 平滑颜色
        double magnitude = sqrt(zx * zx + zy * zy);

        double smooth_iteration =
            iteration + 1.0 - log(log(magnitude)) / log(2.0);

        double t = fmod(smooth_iteration * 0.04, 1.0);

        if (t < 0.4)
        {
            // 深蓝黑 -> 初音青绿
            double k = t / 0.4;

            r = static_cast<unsigned char>(
                10 + k * (57 - 10)
            );

            g = static_cast<unsigned char>(
                18 + k * (197 - 18)
            );

            b = static_cast<unsigned char>(
                28 + k * (187 - 28)
            );
        }
        else if (t < 0.75)
        {
            // 初音青绿 -> 浅青
            double k = (t - 0.4) / 0.35;

            r = static_cast<unsigned char>(
                57 + k * (160 - 57)
            );

            g = static_cast<unsigned char>(
                197 + k * (240 - 197)
            );

            b = static_cast<unsigned char>(
                187 + k * (235 - 187)
            );
        }
        else
        {
            // 浅青 -> 粉色
            double k = (t - 0.75) / 0.25;

            r = static_cast<unsigned char>(
                160 + k * (255 - 160)
            );

            g = static_cast<unsigned char>(
                240 + k * (100 - 240)
            );

            b = static_cast<unsigned char>(
                235 + k * (180 - 235)
            );
        }

        int index = (y * width + x) * 3;

        pixels[index + 0] = r;
        pixels[index + 1] = g;
        pixels[index + 2] = b;
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

    dim3 block_size(8, 8);

    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);


    std::cout << "Grid:" << grid_size.x << "x" << grid_size.y << std::endl;

    std::cout << "Block:" << block_size.x << "x" << block_size.y << std::endl;

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    renderMandelbrot<<<grid_size, block_size>>>(device_pixels, width, height, max_iterations);

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    float kernel_time = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &kernel_time,
        start,
        stop
    ));

    std::cout << "Kernel time:" << kernel_time << " ms" << std::endl;

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
