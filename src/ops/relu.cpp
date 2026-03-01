// ============================================================================
// @file src/ops/relu.cpp
// @brief Hardware-aware implementation of the ReLU activation function.
// @details Utilizes Intel TBB for 1D memory-bound parallelization.
// ============================================================================

#include "ops/relu.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <algorithm> // For std::max
#include <cstdint>

namespace infer {
namespace ops {

template <typename T>
void Relu<T>::forward(const Tensor<T>& input, Tensor<T>& output) const {
    const size_t total_elements = input.size();
    const T* X = input.data();
    T* Y = output.data();

    // ------------------------------------------------------------------------
    // TBB Parallel For (1D)
    // We treat the N-dimensional tensor as a single massive 1D array.
    // TBB divides this contiguous memory block among physical CPU cores.
    // ------------------------------------------------------------------------
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, total_elements),
        [=](const tbb::blocked_range<size_t>& r) {
            
            // Micro-kernel
            for (size_t i = r.begin(); i != r.end(); ++i) {
                Y[i] = std::max(T{0}, X[i]);
            }
        }
    );
}

template <typename T>
void Relu<T>::forward_inplace(Tensor<T>& tensor) const {
    const size_t total_elements = tensor.size();
    T* data = tensor.data();

    // ------------------------------------------------------------------------
    // In-Place Execution
    // Modifies the memory buffer directly. Since we don't write to a separate 
    // 'Y' tensor, we effectively halve the RAM bandwidth bottleneck.
    // ------------------------------------------------------------------------
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, total_elements),
        [=](const tbb::blocked_range<size_t>& r) {
            
            // Micro-kernel
            for (size_t i = r.begin(); i != r.end(); ++i) {
                data[i] = std::max(T{0}, data[i]);
            }
        }
    );
}

// Explicit template instantiations to optimize compile times
template class Relu<float>;
template class Relu<int8_t>;

} // namespace ops
} // namespace infer