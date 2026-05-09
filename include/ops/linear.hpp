// ============================================================================
// @file include/ops/linear.hpp
// @brief Dense (Fully Connected) layer operator.
// ============================================================================

#pragma once

#include "ops/operator_base.hpp"
#include <vector>
#include <stdexcept>
#include <string>

namespace infer {
namespace ops {

template <typename T>
class Linear : public Operator<T> {
private:
    size_t in_features_;
    size_t out_features_;

    Tensor<T> weight_;
    Tensor<T> bias_;

    static constexpr size_t TILE_SIZE_M = 32;
    static constexpr size_t TILE_SIZE_N = 32;
    static constexpr size_t TILE_SIZE_K = 32;

public:
    Linear(size_t in_features, size_t out_features) 
        : in_features_(in_features), out_features_(out_features),
          weight_({out_features, in_features}), bias_({out_features}) {}

    Tensor<T>& weight() { return weight_; }
    Tensor<T>& bias() { return bias_; }

    void forward(const Tensor<T>& input, Tensor<T>& output) const override;
    
    std::vector<size_t> compute_output_shape(const std::span<const size_t>& input_shape) const override {
        return {input_shape[0], out_features_};
    }

    /**
     * @brief Busca y copia sus pesos desde el diccionario global.
     */
    void load_weights(const std::string& prefix, 
                      const std::unordered_map<std::string, Tensor<T>>& state_dict) override;
};

} // namespace ops
} // namespace infer