// Reemplaza el contenido de include/ops/linear.hpp con esto:
#pragma once

#include "ops/operator_base.hpp"
#include <vector>
#include <stdexcept>

namespace infer {
namespace ops {

template <typename T>
class Linear : public Operator<T> {
private:
    size_t in_features_;
    size_t out_features_;

    // ¡La capa ahora es DUEÑA de su memoria (parámetros)!
    Tensor<T> weight_;
    Tensor<T> bias_;

    static constexpr size_t TILE_SIZE_M = 32;
    static constexpr size_t TILE_SIZE_N = 32;
    static constexpr size_t TILE_SIZE_K = 32;

public:
    Linear(size_t in_features, size_t out_features) 
        : in_features_(in_features), out_features_(out_features),
          weight_({out_features, in_features}), bias_({out_features}) {}

    // Exponemos los tensores para poder cargarles los pesos entrenados
    Tensor<T>& weight() { return weight_; }
    Tensor<T>& bias() { return bias_; }

    void forward(const Tensor<T>& input, Tensor<T>& output) const override;
    
    std::vector<size_t> compute_output_shape(const std::span<const size_t>& input_shape) const override {
        return {input_shape[0], out_features_};
    }
};

} // namespace ops
} // namespace infer