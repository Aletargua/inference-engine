// ============================================================================
// @file include/ops/linear.hpp
// @brief Hardware-aware implementation of a Fully Connected (Dense) layer.
// @details Utilizes Cache Blocking (Tiling) and Intel TBB for parallel 
//          execution of the General Matrix Multiply (GEMM) micro-kernels.
// ============================================================================

#pragma once

#include "core/tensor.hpp"
#include <stdexcept>
#include <cassert>

namespace infer {
namespace ops {

/**
 * @brief Represents a Dense (Fully Connected) neural network layer.
 * * Computes Y = X * W^T + b.
 * The weight matrix W is assumed to be already transposed in memory (Row-Major).
 * This "Structure of Arrays" approach ensures that during the dot product,
 * memory reads along the innermost dimension are strictly contiguous,
 * preventing L1/L2 cache misses.
 * * @tparam T The hardware data type (e.g., float, int8_t).
 */
template <typename T>
class Linear {
private:
    size_t in_features_;   /**< Number of input channels (K dimension in GEMM) */
    size_t out_features_;  /**< Number of output channels (N dimension in GEMM) */

    // Hardware-aware tuning parameters
    static constexpr size_t TILE_SIZE_M = 32; /**< Cache block size for Batch dimension */
    static constexpr size_t TILE_SIZE_N = 32; /**< Cache block size for Output dimension */
    static constexpr size_t TILE_SIZE_K = 32; /**< Cache block size for Input dimension */

public:
    /**
     * @brief Constructs a Linear layer configuration.
     * @param in_features Size of each input sample.
     * @param out_features Size of each output sample.
     */
    Linear(size_t in_features, size_t out_features) 
        : in_features_(in_features), out_features_(out_features) {}

    /**
     * @brief Executes the forward pass of the Dense layer (Y = X * W^T + b).
     * * @param input  The input tensor X of shape [Batch, in_features].
     * @param weight The transposed weight tensor W of shape [out_features, in_features].
     * @param bias   The bias tensor b of shape [out_features].
     * @param output The pre-allocated output tensor Y of shape [Batch, out_features].
     * * @note Memory allocation inside this function is strictly forbidden to 
     * guarantee low latency during inference.
     */
    void forward(const Tensor<T>& input, 
                 const Tensor<T>& weight, 
                 const Tensor<T>& bias, 
                 Tensor<T>& output) const;
};

} // namespace ops
} // namespace infer