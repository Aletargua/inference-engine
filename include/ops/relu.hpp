// ============================================================================
// @file include/ops/relu.hpp
// @brief Hardware-aware implementation of the ReLU activation function.
// ============================================================================

#pragma once

#include "core/tensor.hpp"

namespace infer {
namespace ops {

/**
 * @brief Rectified Linear Unit (ReLU) activation function.
 * @details Computes f(x) = max(0, x) element-wise.
 * Supports both out-of-place and zero-copy in-place execution 
 * to maximize memory bandwidth.
 * @tparam T The hardware data type (e.g., float, int8_t).
 */
template <typename T>
class Relu {
public:
    /**
     * @brief Out-of-place forward pass: Y = max(0, X)
     * @param input  The input tensor X.
     * @param output The pre-allocated output tensor Y.
     */
    void forward(const Tensor<T>& input, Tensor<T>& output) const;

    /**
     * @brief In-place forward pass: X = max(0, X)
     * @details Modifies the input tensor directly, saving RAM allocations 
     * and doubling effective memory bandwidth.
     * @param tensor The tensor to modify in-place.
     */
    void forward_inplace(Tensor<T>& tensor) const;
};

} // namespace ops
} // namespace infer