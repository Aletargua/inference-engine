// ============================================================================
// @file tests/test_ops_linear.cpp
// @brief Unit tests for the Dense (Fully Connected) layer operator.
// @details Validates the mathematical correctness of the Y = X * W^T + b 
//          computation using pre-calculated baseline matrices.
// ============================================================================

#include <gtest/gtest.h>
#include "core/tensor.hpp"
#include "ops/linear.hpp"

using namespace infer;
using namespace infer::ops;

/**
 * @brief Test 1: Small 2x2 Matrix Multiplication with Bias.
 * Validates the core GEMM logic and Row-Major memory traversal.
 */
// En tests/test_ops_linear.cpp, actualiza el test ForwardPass2x2:

TEST(LinearOpTest, ForwardPass2x2) {
    const size_t batch_size = 2;
    const size_t in_features = 2;
    const size_t out_features = 2;

    Tensor<float> X({batch_size, in_features});
    Tensor<float> Y({batch_size, out_features}); 

    float* x_ptr = X.data();
    x_ptr[0] = 1.0f; x_ptr[1] = 2.0f;
    x_ptr[2] = 3.0f; x_ptr[3] = 4.0f;

    // Instanciamos la capa PRIMERO
    Linear<float> linear_layer(in_features, out_features);

    // Y luego rellenamos su memoria interna
    float* w_ptr = linear_layer.weight().data();
    w_ptr[0] = 5.0f; w_ptr[1] = 6.0f;
    w_ptr[2] = 7.0f; w_ptr[3] = 8.0f;

    float* b_ptr = linear_layer.bias().data();
    b_ptr[0] = 1.0f; b_ptr[1] = 1.0f;

    // Fíjate qué limpia queda la inferencia ahora:
    linear_layer.forward(X, Y);

    const float* y_ptr = Y.data();
    EXPECT_FLOAT_EQ(y_ptr[0], 18.0f);
    EXPECT_FLOAT_EQ(y_ptr[1], 24.0f);
    EXPECT_FLOAT_EQ(y_ptr[2], 40.0f);
    EXPECT_FLOAT_EQ(y_ptr[3], 54.0f);
}