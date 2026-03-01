# High-Performance ML Inference Engine (C++23)

![C++23](https://img.shields.io/badge/C++-23-blue.svg)


> ** [Read the full Architecture & Design Document (PDF)](./Inference.Engine.pdf)** > *A comprehensive dive into the theoretical foundations, cache-blocking mathematics, and hardware-aware C++ implementation.*

A high-performance, hardware-aware Machine Learning inference engine built entirely from scratch in C++23. This project aims to demystify the "black box" of AI infrastructure by providing a transparent, highly optimized CPU backend for executing neural networks.

## Architecture Overview

The codebase strictly follows the Separation of Concerns (SoC) principle, mirroring industry-standard frameworks like PyTorch:

* **`core/` (Memory Management):** Hardware-friendly `Tensor` class mapped to 1D contiguous RAM arrays. Implements Row-Major (C-contiguous) layouts with strides for optimal SIMD prefetching.
* **`ops/` (Compute Kernels):** Heavy mathematical operations. Features stateful, polymorphic layers inheriting from a base `Operator` class.
* **`graph/` (Orchestration):** The `Sequential` model orchestrator handles automated shape inference and intermediate memory management during the forward pass.

## Key Hardware Optimizations

* **Cache Blocking (Tiling):** The Dense (`Linear`) layer partitions matrices into L1-cache-friendly 32x32 tiles, drastically reducing DDR RAM roundtrips and maximizing Arithmetic Intensity.
* **Parallel Execution:** Leverages **Intel Threading Building Blocks (TBB)** for lock-free, task-based multithreading across all physical CPU cores.
* **In-Place Operations:** Memory-bound operations like `ReLU` utilize zero-copy, in-place memory modifications to halve memory bandwidth usage.
* **Structure of Arrays (SOA):** Weight matrices are transposed in memory to guarantee sequential memory reads during dot products, keeping the CPU hardware prefetcher saturated.

## Build Instructions (Dockerized)

To guarantee portability, the build environment is fully containerized.

```bash
# 1. Build the Docker image
docker build -t inference_engine:v1 .

# 2. Run the interactive container
docker run -it --rm -v $(pwd):/workspace inference_engine:v1

# 3. Inside the container: Compile and Test
cmake -B build
cmake --build build -j $(nproc)
cd build
ctest --output-on-failure
