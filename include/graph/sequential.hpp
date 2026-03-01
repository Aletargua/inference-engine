// ============================================================================
// @file include/graph/sequential.hpp
// @brief Orchestrates a linear stack of neural network layers.
// ============================================================================

#pragma once

#include "ops/operator_base.hpp"
#include "core/tensor.hpp"
#include <vector>
#include <memory>

namespace infer {
namespace graph {

/**
 * @brief A Sequential container (similar to PyTorch's nn.Sequential).
 * @details Modules will be added to it in the order they are passed.
 * It acts as the orchestrator of the forward pass, managing intermediate memory.
 */
template <typename T>
class Sequential {
private:
    std::vector<std::unique_ptr<ops::Operator<T>>> layers_;

public:
    Sequential() = default;

    /**
     * @brief Adds a new layer to the sequential stack.
     * @param layer A unique pointer to an Operator (Type Erasure).
     */
    void add_layer(std::unique_ptr<ops::Operator<T>> layer);

    /**
     * @brief Executes the entire stack of layers sequentially.
     * @param input The initial input tensor.
     * @return Tensor<T> The final output tensor of the network.
     */
    Tensor<T> forward(const Tensor<T>& input) const;
};

} // namespace graph
} // namespace infer