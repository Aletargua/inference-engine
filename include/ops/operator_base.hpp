// ============================================================================
// @file include/ops/operator_base.hpp
// @brief Pure virtual interface for all neural network operations.
// ============================================================================

#pragma once
#include "core/tensor.hpp"
#include <span>
#include <vector>
#include <string>
#include <unordered_map>

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

    /**
     * @brief Injects pre-trained weights from a global state dictionary.
     * @details Stateless layers (like ReLU) inherit this empty implementation.
     * Stateful layers (like Linear) override it to extract their parameters.
     *
     * @param prefix The layer's namespace prefix (e.g., "layer0").
     * @param state_dict Map of tensor names to allocated Tensors.
     */
    virtual void load_weights(const std::string& /*prefix*/, 
                              const std::unordered_map<std::string, Tensor<T>>& /*state_dict*/) {
        // Al comentar los nombres de las variables silenciamos el -Wunused-parameter
        // manteniendo intacto el contrato polimórfico.
    }
};

} // namespace ops
} // namespace infer