cmake_minimum_required(VERSION 3.24)
project(CUDA_PLAY LANGUAGES CXX CUDA)

# RTX 3080 是 Ampere 架构，计算能力为 8.6。仍允许通过命令行覆盖该值：
# cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(CUDA_PLAY main.cu)
target_link_libraries(CUDA_PLAY PRIVATE CUDA::cudart)

set_target_properties(CUDA_PLAY PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(CUDA_PLAY PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
