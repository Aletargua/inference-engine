#pragma once

#include <vector>
#include <span>
#include <numeric>
#include <stdexcept>
#include <initializer_list>

namespace infer {

/**
 * @brief A multi-dimensional array mapped to contiguous linear memory.
 * * Similar to how a Structure of Arrays (SOA) flattens 3D coordinates into a 1D buffer 
 * in a Graphics Renderer, this class maps N-dimensional tensors to a 1D `std::vector`.
 * It manages hardware-friendly memory layouts using strides.
 * * @tparam T The primitive data type of the tensor elements (e.g., float, int8_t).
 */
template <typename T>
class Tensor {
private:
    std::vector<T> data_;             /**< Contiguous linear memory buffer. */
    std::vector<size_t> shape_;       /**< Dimensions of the tensor (e.g., {Batch, Channels, Height, Width}). */
    std::vector<size_t> strides_;     /**< Memory jumps required to move along each dimension. */

    /**
     * @brief Computes the strides for a Row-Major (C-contiguous) memory layout.
     * * This layout ensures that the innermost dimension elements are adjacent in RAM,
     * maximizing L1/L2 cache hits and allowing SIMD hardware prefetchers to work efficiently.
     */
    void compute_strides() {
        strides_.resize(shape_.size());
        size_t current_stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            strides_[i] = current_stride;
            current_stride *= shape_[i];
        }
    }

public:
    /**
     * @brief Constructs a Tensor with a specific shape.
     * * Allocates contiguous memory and initializes it to zero.
     * * @param shape A list representing the size of each dimension.
     */
    Tensor(std::initializer_list<size_t> shape) : shape_(shape) {
        size_t total_elements = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
        data_.resize(total_elements, T{0});
        compute_strides();
    }

    /**
     * @brief Gets a raw pointer to the underlying contiguous memory.
     * * Essential for passing data to highly optimized SIMD routines or Intel TBB tasks.
     * * @return T* Pointer to the first element.
     */
    T* data() { return data_.data(); }

    /**
     * @brief Gets a read-only raw pointer to the underlying contiguous memory.
     * * @return const T* Pointer to the first element.
     */
    const T* data() const { return data_.data(); }

    /**
     * @brief Returns the dimensions of the tensor.
     * * @return std::span<const size_t> A lightweight, non-owning view of the shape array.
     */
    std::span<const size_t> shape() const { return shape_; }

    /**
     * @brief Returns the computed memory strides.
     * * @return std::span<const size_t> A lightweight, non-owning view of the strides array.
     */
    std::span<const size_t> strides() const { return strides_; }

    /**
     * @brief Returns the total number of elements allocated in memory.
     * * @return size_t Total element count.
     */
    size_t size() const { return data_.size(); }
};

} // namespace infer