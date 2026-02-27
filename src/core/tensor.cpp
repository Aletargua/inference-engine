// ============================================================================
// @file src/core/tensor.cpp
// @brief Explicit template instantiations for the Tensor class.
// @details In C++, template implementations typically reside in header files. 
//          However, to significantly reduce compilation times in large codebases 
//          and hide implementation details where possible, we explicitly 
//          instantiate the Tensor class for the hardware data types we support.
// ============================================================================

#include "core/tensor.hpp"
#include <cstdint>

namespace infer {

// ----------------------------------------------------------------------------
// Explicit Template Instantiation
// Instructs the compiler to pre-compile the Tensor class for specific types,
// creating binary objects (.o) for these versions inside 'infer_core'.
// ----------------------------------------------------------------------------

/**
 * @brief Standard 32-bit floating-point tensor.
 * Used for standard precision inference and unquantized weights.
 */
template class Tensor<float>;

/**
 * @brief 8-bit signed integer tensor.
 * Crucial for Edge Computing scenarios involving quantized models (INT8),
 * maximizing memory bandwidth and SIMD throughput.
 */
template class Tensor<int8_t>;

// If we need FP16 or BF16 in the future (e.g., using specialized hardware types),
// we will add their explicit instantiations here.

} // namespace infer