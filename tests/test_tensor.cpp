// ============================================================================
// @file tests/test_tensor.cpp
// @brief Unit tests for the Tensor class, validating memory layout and strides.
// @details Employs Test-Driven Development (TDD) using GoogleTest to ensure
//          the mathematical abstraction maps correctly to 1D RAM for optimal
//          hardware cache utilization.
// ============================================================================

#include <gtest/gtest.h>
#include "core/tensor.hpp"

using namespace infer;

/**
 * @brief Test 1: 1D Tensor (Vector)
 * Validates basic memory allocation and 1D contiguous strides.
 */
TEST(TensorTest, VectorProperties) {
    Tensor<float> vec({10});
    
    // Verify total elements allocated in the contiguous buffer
    EXPECT_EQ(vec.size(), 10);
    EXPECT_EQ(vec.shape().size(), 1);
    EXPECT_EQ(vec.shape()[0], 10);
    
    // In a 1D tensor, the step to the next element is strictly 1 (contiguous in RAM)
    EXPECT_EQ(vec.strides()[0], 1); 
}

/**
 * @brief Test 2: 2D Tensor (Matrix)
 * Validates row-major (C-contiguous) layout. 
 * Crucial for GEMM (General Matrix Multiply) cache performance.
 */
TEST(TensorTest, MatrixProperties) {
    // Shape: 4 rows, 5 columns
    Tensor<float> mat({4, 5});
    
    EXPECT_EQ(mat.size(), 20);
    
    // Strides calculation for row-major:
    // stride[0] = shape[1] = 5 (Jump 5 elements to reach the same column in the next row)
    // stride[1] = 1 (Elements in the same row are adjacent in L1 cache)
    EXPECT_EQ(mat.strides()[0], 5);
    EXPECT_EQ(mat.strides()[1], 1);
}

/**
 * @brief Test 3: 4D Tensor (NCHW Layout)
 * Validates complex strides used in Convolutional Neural Networks (CNNs).
 * This maps directly to a Structure of Arrays (SOA) approach.
 */
TEST(TensorTest, NCHWStrides) {
    // Shape: Batch=2, Channels=3, Height=4, Width=5
    Tensor<float> img({2, 3, 4, 5});
    
    EXPECT_EQ(img.size(), 120);
    
    // Expected strides for hardware-friendly contiguous memory:
    // stride[3] (Width)    = 1 (SIMD vectorization target)
    // stride[2] (Height)   = shape[3] = 5
    // stride[1] (Channels) = shape[2] * shape[3] = 20
    // stride[0] (Batch)    = shape[1] * shape[2] * shape[3] = 60
    EXPECT_EQ(img.strides()[0], 60);
    EXPECT_EQ(img.strides()[1], 20);
    EXPECT_EQ(img.strides()[2], 5);
    EXPECT_EQ(img.strides()[3], 1);
}