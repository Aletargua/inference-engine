// ============================================================================
// @file src/io/model_loader.cpp
// @brief Implementation of the Simple Binary Standard (SBS) model loader.
// ============================================================================

#include "io/model_loader.hpp"
#include <fstream>
#include <vector>

namespace infer {
namespace io {

template <typename T>
std::unordered_map<std::string, Tensor<T>> ModelLoader::load(const std::string& filepath) {
    // 1. Abrir archivo en modo lectura binaria
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("ModelLoader Error: Could not open file " + filepath);
    }

    // 2. Leer y validar el Magic Number ('INF1')
    char magic[4];
    file.read(magic, 4);
    if (std::string(magic, 4) != "INF1") {
        throw std::runtime_error("ModelLoader Error: Invalid file format. Expected 'INF1' magic number.");
    }

    // 3. Leer la cantidad de tensores que hay en el archivo
    uint32_t num_tensors;
    file.read(reinterpret_cast<char*>(&num_tensors), sizeof(uint32_t));

    std::unordered_map<std::string, Tensor<T>> result_map;

    // 4. Bucle para leer cada tensor
    for (uint32_t i = 0; i < num_tensors; ++i) {
        
        // A) Leer el nombre del tensor (ej: "layer1.weight")
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        // B) Leer la forma (Shape)
        uint32_t rank;
        file.read(reinterpret_cast<char*>(&rank), sizeof(uint32_t));
        
        std::vector<size_t> shape(rank);
        for (uint32_t r = 0; r < rank; ++r) {
            uint32_t dim;
            file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
            shape[r] = dim;
        }

        // C) Crear el Tensor y leer los datos crudos de golpe (Zero-Copy-ish)
        Tensor<T> tensor(shape);
        size_t bytes_to_read = tensor.size() * sizeof(T);
        file.read(reinterpret_cast<char*>(tensor.data()), bytes_to_read);

        // D) Guardarlo en el diccionario
        result_map.emplace(name, std::move(tensor));
    }

    return result_map;
}

// ============================================================================
// Explicit template instantiations
// Required because the implementation is in a .cpp file.
// ============================================================================
template std::unordered_map<std::string, Tensor<float>> ModelLoader::load<float>(const std::string&);
template std::unordered_map<std::string, Tensor<int8_t>> ModelLoader::load<int8_t>(const std::string&);

} // namespace io
} // namespace infer