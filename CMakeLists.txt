# =============================================================================
# @file CMakeLists.txt (Root)
# @brief Master CMake configuration for the High-Performance Inference Engine.
# @details Defines global C++23 standards, compiler optimizations for 
#          hardware-aware execution (SIMD), and orchestrates subdirectories.
# =============================================================================

cmake_minimum_required(VERSION 3.22)
project(InferenceEngine VERSION 0.1.0 LANGUAGES CXX)

# =============================================================================
# 1. C++ Standard Configuration (Software Quality)
# @brief Enforce strict C++23 standard without compiler-specific extensions 
#        to guarantee portability across different hardware environments.
# =============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# 2. Compiler Optimization Flags (Hardware-Awareness & Rigor)
# @brief Flags to maximize CPU throughput and maintain strict code quality.
# @note -march=native is CRITICAL. It instructs GCC/Clang to analyze the host 
#       CPU and emit vectorized SIMD instructions (AVX2, AVX-512) for our math 
#       kernels. Without this, the engine will bottleneck on memory bandwidth.
# =============================================================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -Wpedantic")

# =============================================================================
# 3. Dependency Resolution
# @brief Locate Intel TBB for task-based parallelism and GoogleTest for TDD.
# @note These packages are assumed to be provided by the Docker environment.
# =============================================================================
find_package(TBB REQUIRED)
find_package(GTest REQUIRED)

# =============================================================================
# 4. Global Test Enablement
# @brief Allows the use of 'ctest' command at the project root to run all 
#        discovered unit tests in the /tests subdirectory.
# =============================================================================
enable_testing()

# =============================================================================
# 5. Modular Subdirectories Architecture
# @brief Delegate target creation (libraries, executables) to specialized folders.
#        This promotes a decoupled, highly scalable codebase.
# =============================================================================
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(tests)