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
TEST(LinearOpTest, ForwardPass2x2) {
    // 1. Define shapes
    const size_t batch_size = 2;
    const size_t in_features = 2;
    const size_t out_features = 2;

    // 2. Allocate tensors
    Tensor<float> X({batch_size, in_features});
    Tensor<float> W({out_features, in_features}); // Stored transposed for SOA/SIMD benefits
    Tensor<float> b({out_features});
    Tensor<float> Y({batch_size, out_features});  // Output buffer

    // 3. Populate input X: [[1, 2], [3, 4]]
    float* x_ptr = X.data();
    x_ptr[0] = 1.0f; x_ptr[1] = 2.0f;
    x_ptr[2] = 3.0f; x_ptr[3] = 4.0f;

    // 4. Populate weights W: [[5, 6], [7, 8]]
    float* w_ptr = W.data();
    w_ptr[0] = 5.0f; w_ptr[1] = 6.0f;
    w_ptr[2] = 7.0f; w_ptr[3] = 8.0f;

    // 5. Populate bias b: [1, 1]
    float* b_ptr = b.data();
    b_ptr[0] = 1.0f; b_ptr[1] = 1.0f;

    // 6. Execute operator
    Linear<float> linear_layer(in_features, out_features);
    linear_layer.forward(X, W, b, Y);

    // 7. Validate results against manual mathematical calculation
    const float* y_ptr = Y.data();
    EXPECT_FLOAT_EQ(y_ptr[0], 18.0f);
    EXPECT_FLOAT_EQ(y_ptr[1], 24.0f);
    EXPECT_FLOAT_EQ(y_ptr[2], 40.0f);
    EXPECT_FLOAT_EQ(y_ptr[3], 54.0f);
}