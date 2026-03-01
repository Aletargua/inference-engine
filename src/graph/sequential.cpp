// ============================================================================
// @file src/graph/sequential.cpp
// @brief Implementation of the Sequential graph orchestrator.
// ============================================================================

#include "graph/sequential.hpp"
#include <stdexcept>

namespace infer {
namespace graph {

template <typename T>
void Sequential<T>::add_layer(std::unique_ptr<ops::Operator<T>> layer) {
    layers_.push_back(std::move(layer));
}

template <typename T>
Tensor<T> Sequential<T>::forward(const Tensor<T>& input) const {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential graph has no layers to execute.");
    }

    // Puntero que apuntará al tensor de entrada de la capa actual
    const Tensor<T>* current_input = &input;
    
    // Contenedor para almacenar y liberar automáticamente los tensores intermedios
    std::unique_ptr<Tensor<T>> intermediate_storage;

    for (size_t i = 0; i < layers_.size(); ++i) {
        
        // 1. Inferencia de forma: ¿Qué tamaño de RAM necesita esta capa para su salida?
        std::vector<size_t> out_shape = layers_[i]->compute_output_shape(current_input->shape());
        
        // 2. Reservar la RAM para el resultado
        auto output_tensor = std::make_unique<Tensor<T>>(out_shape);
        
        // 3. Ejecutar la matemática de la capa (GEMM o ReLU)
        layers_[i]->forward(*current_input, *output_tensor);
        
        // 4. El tensor de salida se convierte en la entrada de la siguiente capa
        intermediate_storage = std::move(output_tensor);
        current_input = intermediate_storage.get();
    }

    // Retorna una copia de seguridad del último tensor antes de destruir la memoria intermedia
    return *current_input;
}

// Explicit template instantiations
template class Sequential<float>;
template class Sequential<int8_t>;

} // namespace graph
} // namespace infer