// ============================================================================
// @file src/graph/sequential.cpp
// @brief Implementation of the Sequential graph orchestrator.
// ============================================================================

#include "graph/sequential.hpp"
#include <stdexcept>

namespace infer {
namespace graph {

template <typename T>
void Sequential<T>::add_layer(std::unique_ptr<ops::Operator<T>> layer, const std::string& custom_prefix) {
    layers_.push_back(std::move(layer));
    if (custom_prefix.empty()) {
        // Asignación automática: "layer_0", "layer_1"...
        layer_prefixes_.push_back("layer_" + std::to_string(layers_.size() - 1));
    } else {
        layer_prefixes_.push_back(custom_prefix);
    }
}

template <typename T>
void Sequential<T>::load_state_dict(const std::unordered_map<std::string, Tensor<T>>& state_dict) {
    // Propagación polimórfica (Broadcast)
    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i]->load_weights(layer_prefixes_[i], state_dict);
    }
}

template <typename T>
Tensor<T> Sequential<T>::forward(const Tensor<T>& input) const {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential graph has no layers to execute.");
    }

    const Tensor<T>* current_input = &input;
    std::unique_ptr<Tensor<T>> intermediate_storage;

    for (size_t i = 0; i < layers_.size(); ++i) {
        std::vector<size_t> out_shape = layers_[i]->compute_output_shape(current_input->shape());
        auto output_tensor = std::make_unique<Tensor<T>>(out_shape);
        
        layers_[i]->forward(*current_input, *output_tensor);
        
        intermediate_storage = std::move(output_tensor);
        current_input = intermediate_storage.get();
    }

    return *current_input;
}

// Explicit template instantiations
template class Sequential<float>;
template class Sequential<int8_t>;

} // namespace graph
} // namespace infer