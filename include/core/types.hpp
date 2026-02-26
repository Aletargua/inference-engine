#pragma once

#include <cstdint>

namespace infer {

/**
 * @brief Represents the underlying hardware data types supported by the engine.
 * * In high-performance Machine Learning, choosing the right data type is crucial
 * for maximizing memory bandwidth and SIMD vectorization.
 */
enum class DType {
    Float32,    /**< Standard 32-bit floating point. Best for precision. */
    Int8        /**< 8-bit integer. Used for quantized, bandwidth-bound inference. */
};

} // namespace infer