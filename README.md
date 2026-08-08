# CUDA_PLAY

面向 CUDA C++ 学习的最小工程，当前示例会在 GPU 上完成向量加法并校验结果。

## 命令行构建

```bash
cmake --preset debug
cmake --build --preset debug
./build/debug/CUDA_PLAY
```

发布模式：

```bash
cmake --preset release
cmake --build --preset release
./build/release/CUDA_PLAY
```

检查 kernel 的内存访问错误：

```bash
compute-sanitizer ./build/debug/CUDA_PLAY
```

## CLion

重新加载 CMake 项目后，选择 `CUDA Debug` preset 和 `CUDA_PLAY` 运行目标即可。
工具链使用 `/usr/local/cuda/bin/nvcc`，目标架构为 RTX 3080 对应的 `sm_86`。

## 已安装工具

- `nvcc`：CUDA C++ 编译器
- `cuda-gdb`：CUDA 调试器
- `compute-sanitizer`：内存、同步和竞争检查
- `ncu` / Nsight Compute：kernel 性能分析
- `nsys` / Nsight Systems：CPU/GPU 时间线分析
