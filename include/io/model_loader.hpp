// ============================================================================
// @file include/io/model_loader.hpp
// @brief Parses Simple Binary Standard (SBS) files directly into Tensors.
// ============================================================================

#pragma once

#include "core/tensor.hpp"
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>

namespace infer {
namespace io {

/**
 * @brief Handles reading and parsing of model weights from the disk.
 * @details Reads the custom 'INF1' binary format, extracting structural metadata
 * and loading raw byte blocks directly into contiguous Tensor memory 
 * to maximize loading speed and minimize RAM fragmentation.
 */
class ModelLoader {
public:
    /**
     * @brief Loads a state dictionary from an SBS binary file.
     * * @tparam T The hardware data type (typically float).
     * @param filepath The absolute or relative path to the .bin file.
     * @return std::unordered_map<std::string, Tensor<T>> A map associating 
     * the tensor's name (e.g., "layer1.weight") to the allocated Tensor.
     * @throws std::runtime_error If the file cannot be found, if the magic number 
     * is invalid, or if the file is corrupted.
     */
    template <typename T>
    static std::unordered_map<std::string, Tensor<T>> load(const std::string& filepath);
};

} // namespace io
} // namespace infer