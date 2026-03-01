// ============================================================================
// @file tests/test_ops_relu.cpp
// @brief Unit tests for the ReLU activation operator.
// ============================================================================

#include <gtest/gtest.h>
#include "core/tensor.hpp"
#include "ops/relu.hpp"

using namespace infer;
using namespace infer::ops;

TEST(ReluOpTest, OutOfPlaceForward) {
    Tensor<float> X({4});
    Tensor<float> Y({4});

    float* x_ptr = X.data();
    x_ptr[0] = -1.5f; x_ptr[1] = 0.0f; 
    x_ptr[2] = 2.5f;  x_ptr[3] = -10.0f;

    Relu<float> relu;
    relu.forward(X, Y);

    const float* y_ptr = Y.data();
    EXPECT_FLOAT_EQ(y_ptr[0], 0.0f);
    EXPECT_FLOAT_EQ(y_ptr[1], 0.0f);
    EXPECT_FLOAT_EQ(y_ptr[2], 2.5f);
    EXPECT_FLOAT_EQ(y_ptr[3], 0.0f);
}

TEST(ReluOpTest, InPlaceForward) {
    Tensor<float> X({4});
    
    float* x_ptr = X.data();
    x_ptr[0] = -5.0f; x_ptr[1] = 3.14f; 
    x_ptr[2] = -0.1f; x_ptr[3] = 42.0f;

    Relu<float> relu;
    relu.forward_inplace(X);

    EXPECT_FLOAT_EQ(x_ptr[0], 0.0f);
    EXPECT_FLOAT_EQ(x_ptr[1], 3.14f);
    EXPECT_FLOAT_EQ(x_ptr[2], 0.0f);
    EXPECT_FLOAT_EQ(x_ptr[3], 42.0f);
}