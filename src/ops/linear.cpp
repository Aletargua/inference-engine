// ============================================================================
// @file src/ops/linear.cpp
// @brief Hardware-aware implementation of the Dense layer using Cache Tiling
//        and Intel Threading Building Blocks (TBB).
// ============================================================================

#include "ops/linear.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <algorithm> // For std::min
#include <cstring>   // For std::memcpy

namespace infer {
namespace ops {

template <typename T>
void Linear<T>::forward(const Tensor<T>& input, Tensor<T>& output) const {
    
    const size_t batch_size = input.shape()[0];
    const T* X = input.data();
    
    const T* W = weight_.data();
    const T* b = bias_.data();
    
    T* Y = output.data();

    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(
            0, batch_size, TILE_SIZE_M,      // Rows dimension (Batch)
            0, out_features_, TILE_SIZE_N    // Cols dimension (Output Features)
        ),
        [=, this](const tbb::blocked_range2d<size_t>& r) {
            for (size_t i = r.rows().begin(); i != r.rows().end(); ++i) {
                for (size_t j = r.cols().begin(); j != r.cols().end(); ++j) {
                    
                    T sum = b[j];
                    
                    for (size_t k_block = 0; k_block < in_features_; k_block += TILE_SIZE_K) {
                        size_t k_end = std::min(k_block + TILE_SIZE_K, in_features_);
                        
                        for (size_t k = k_block; k < k_end; ++k) {
                            sum += X[i * in_features_ + k] * W[j * in_features_ + k];
                        }
                    }
                    
                    Y[i * out_features_ + j] = sum;
                }
            }
        }
    );
}

// ----------------------------------------------------------------------------
// Weight Injection
// ----------------------------------------------------------------------------
template <typename T>
void Linear<T>::load_weights(const std::string& prefix, 
                             const std::unordered_map<std::string, Tensor<T>>& state_dict) {
    
    std::string w_name = prefix + ".weight";
    std::string b_name = prefix + ".bias";

    // 1. Cargar la matriz de pesos (Weights)
    auto w_it = state_dict.find(w_name);
    if (w_it != state_dict.end()) {
        const Tensor<T>& loaded_w = w_it->second;
        if (loaded_w.size() != weight_.size()) {
            throw std::runtime_error("Linear Layer Error: Shape mismatch for " + w_name);
        }
        std::memcpy(weight_.data(), loaded_w.data(), weight_.size() * sizeof(T));
    } else {
        throw std::runtime_error("Linear Layer Error: Missing weight tensor in state_dict: " + w_name);
    }

    // 2. Cargar el vector de sesgo (Bias)
    auto b_it = state_dict.find(b_name);
    if (b_it != state_dict.end()) {
        const Tensor<T>& loaded_b = b_it->second;
        if (loaded_b.size() != bias_.size()) {
            throw std::runtime_error("Linear Layer Error: Shape mismatch for " + b_name);
        }
        std::memcpy(bias_.data(), loaded_b.data(), bias_.size() * sizeof(T));
    } else {
        throw std::runtime_error("Linear Layer Error: Missing bias tensor in state_dict: " + b_name);
    }
}

// Explicit template instantiations
template class Linear<float>;
template class Linear<int8_t>;

} // namespace ops
} // namespace infer