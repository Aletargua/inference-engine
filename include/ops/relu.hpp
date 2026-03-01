// ============================================================================
// @file include/ops/relu.hpp
// @brief Hardware-aware implementation of the ReLU activation function.
// ============================================================================

#pragma once

#include "ops/operator_base.hpp"
#include <vector>

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
class Relu : public Operator<T> {
public:
    /**
     * @brief Out-of-place forward pass: Y = max(0, X)
     * @param input  The input tensor X.
     * @param output The pre-allocated output tensor Y.
     */
    void forward(const Tensor<T>& input, Tensor<T>& output) const override;

    /**
     * @brief In-place forward pass: X = max(0, X)
     * @details Modifies the input tensor directly, saving RAM allocations 
     * and doubling effective memory bandwidth.
     * @param tensor The tensor to modify in-place.
     */
    void forward_inplace(Tensor<T>& tensor) const;

    /**
     * @brief Shape inference: ReLU does not change the shape of the tensor.
     */
    std::vector<size_t> compute_output_shape(const std::span<const size_t>& input_shape) const override {
        // Return a copy of the input shape
        return std::vector<size_t>(input_shape.begin(), input_shape.end());
    }
};

} // namespace ops
} // namespace infer