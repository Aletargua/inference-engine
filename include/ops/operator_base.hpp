// ============================================================================
// @file include/ops/operator_base.hpp
// @brief Pure virtual interface for all neural network operations.
// ============================================================================

#pragma once
#include "core/tensor.hpp"
#include <span>
#include <vector>

namespace infer {
namespace ops {

template <typename T>
class Operator {
public:
    virtual ~Operator() = default;

    /**
     * @brief Universal contract: One tensor in, one tensor out.
     */
    virtual void forward(const Tensor<T>& input, Tensor<T>& output) const = 0;

    /**
     * @brief Shape inference: Crucial for Graph memory pre-allocation.
     */
    virtual std::vector<size_t> compute_output_shape(const std::span<const size_t>& input_shape) const = 0;
};

} // namespace ops
} // namespace infer