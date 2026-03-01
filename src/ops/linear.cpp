// ============================================================================
// @file src/ops/linear.cpp
// @brief Hardware-aware implementation of the Dense layer using Cache Tiling
//        and Intel Threading Building Blocks (TBB).
// ============================================================================

#include "ops/linear.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <algorithm> // For std::min

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

    // ------------------------------------------------------------------------
    // TBB Parallel For with 2D Blocked Range
    // We partition the output matrix Y into chunks (tiles) of size M x N.
    // TBB will automatically distribute these tiles across physical CPU cores.
    // ------------------------------------------------------------------------
    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(
            0, batch_size, TILE_SIZE_M,      // Rows dimension (Batch)
            0, out_features_, TILE_SIZE_N    // Cols dimension (Output Features)
        ),
        [=, this](const tbb::blocked_range2d<size_t>& r) {
            // This lambda function executes on a single CPU core for a specific tile.
            
            // Loop over the assigned Tile bounds for Batch (M)
            for (size_t i = r.rows().begin(); i != r.rows().end(); ++i) {
                // Loop over the assigned Tile bounds for Output Features (N)
                for (size_t j = r.cols().begin(); j != r.cols().end(); ++j) {
                    
                    // 1. Initialize accumulator with bias for this specific element
                    T sum = b[j];
                    
                    // --------------------------------------------------------
                    // CACHE BLOCKING (TILING) FOR THE 'K' DIMENSION
                    // Instead of reading the entire row of X (which causes L1 evictions),
                    // we process the dot-product in chunks of TILE_SIZE_K.
                    // --------------------------------------------------------
                    for (size_t k_block = 0; k_block < in_features_; k_block += TILE_SIZE_K) {
                        
                        // Calculate how far we can go without going out of bounds
                        size_t k_end = std::min(k_block + TILE_SIZE_K, in_features_);
                        
                        // Micro-kernel: The inner dot-product loop
                        // This loop operates purely on data that is now guaranteed 
                        // to be hot in the L1/L2 cache.
                        for (size_t k = k_block; k < k_end; ++k) {
                            // Because W is transposed (SOA), W[j * in_features_ + k]
                            // is a sequential memory read. Hardware prefetcher loves this.
                            sum += X[i * in_features_ + k] * W[j * in_features_ + k];
                        }
                    }
                    
                    // 3. Write final result directly to RAM (Avoids register spilling)
                    Y[i * out_features_ + j] = sum;
                }
            }
        }
    );
}

// Explicit template instantiations
template class Linear<float>;
template class Linear<int8_t>;

} // namespace ops
} // namespace infer