// ============================================================================
// @file include/graph/sequential.hpp
// @brief Orchestrates a linear stack of neural network layers.
// ============================================================================

#pragma once

#include "ops/operator_base.hpp"
#include "core/tensor.hpp"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace infer {
namespace graph {

template <typename T>
class Sequential {
private:
    std::vector<std::unique_ptr<ops::Operator<T>>> layers_;
    std::vector<std::string> layer_prefixes_; /**< Almacena el prefijo de cada capa */

public:
    Sequential() = default;

    /**
     * @brief Adds a new layer to the sequential stack with an automatic or custom prefix.
     * @param layer A unique pointer to an Operator.
     * @param custom_prefix Optional prefix (e.g. "hidden"). Defaults to "layer_X".
     */
    void add_layer(std::unique_ptr<ops::Operator<T>> layer, const std::string& custom_prefix = "");

    /**
     * @brief Propaga el diccionario de pesos a todas las capas registradas.
     */
    void load_state_dict(const std::unordered_map<std::string, Tensor<T>>& state_dict);

    Tensor<T> forward(const Tensor<T>& input) const;
};

} // namespace graph
} // namespace infer