// ============================================================================
// @file src/ops/linear.cpp
// @brief Implementation of the Dense layer micro-kernels.
// ============================================================================

#include "ops/linear.hpp"
#include <cstdint>

namespace infer {
namespace ops {

template <typename T>
void Linear<T>::forward(const Tensor<T>& input, 
                        const Tensor<T>& weight, 
                        const Tensor<T>& bias, 
                        Tensor<T>& output) const {
    
    const size_t batch_size = input.shape()[0];
    const T* X = input.data();
    const T* W = weight.data();
    const T* b = bias.data();
    T* Y = output.data();

    // Naïve GEMM: Y = X * W^T + b
    // Complexity: O(Batch * OutFeatures * InFeatures)
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_features_; ++j) {
            
            // 1. Initialize accumulator with bias
            T sum = b[j];
            
            // 2. Dot product between X's row and W's row (W is transposed)
            for (size_t k = 0; k < in_features_; ++k) {
                // Offset calculation for 1D RAM mapping:
                // X_val = X[i * in_features + k]
                // W_val = W[j * in_features + k]  <-- Sequential memory read! (SOA)
                sum += X[i * in_features_ + k] * W[j * in_features_ + k];
            }
            
            // 3. Write to output buffer
            Y[i * out_features_ + j] = sum;
        }
    }
}

// Explicit template instantiations to optimize compile times
template class Linear<float>;
template class Linear<int8_t>;

} // namespace ops
} // namespace infer