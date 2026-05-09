// ============================================================================
// @file tests/test_graph_sequential.cpp
// @brief Integration test for the Sequential orchestrator (Mini-MLP).
// ============================================================================

#include <gtest/gtest.h>
#include "core/tensor.hpp"
#include "ops/linear.hpp"
#include "ops/relu.hpp"
#include "graph/sequential.hpp"
#include "io/model_loader.hpp"
#include <fstream>
#include <cstdio>

using namespace infer;
using namespace infer::ops;
using namespace infer::graph;

TEST(SequentialGraphTest, MLPForwardPass) {
    // 1. Inicializar el Modelo
    Sequential<float> model;

    // 2. Capa Oculta (2 entradas -> 3 neuronas)
    auto layer1 = std::make_unique<Linear<float>>(2, 3);
    float* w1 = layer1->weight().data();
    w1[0] = 1.0f; w1[1] = 0.5f;   // Neurona 0
    w1[2] = -1.0f; w1[3] = 2.0f;  // Neurona 1
    w1[4] = 0.0f; w1[5] = -0.5f;  // Neurona 2
    
    float* b1 = layer1->bias().data();
    b1[0] = 0.1f; b1[1] = 0.2f; b1[2] = 0.3f;
    model.add_layer(std::move(layer1));

    // 3. Activación (No-Linealidad)
    model.add_layer(std::make_unique<Relu<float>>());

    // 4. Capa de Salida (3 neuronas -> 1 salida)
    auto layer2 = std::make_unique<Linear<float>>(3, 1);
    float* w2 = layer2->weight().data();
    w2[0] = 1.0f; w2[1] = 1.0f; w2[2] = 1.0f;
    
    float* b2 = layer2->bias().data();
    b2[0] = 0.0f;
    model.add_layer(std::move(layer2));

    // 5. Crear los datos de entrada de prueba (Batch=1, Features=2)
    Tensor<float> X({1, 2});
    X.data()[0] = 2.0f; X.data()[1] = 4.0f;

    // 6. ¡¡EJECUTAR LA RED NEURONAL!!
    Tensor<float> Y = model.forward(X);

    // 7. Validar la salida
    ASSERT_EQ(Y.shape().size(), 2);
    EXPECT_EQ(Y.shape()[0], 1);
    EXPECT_EQ(Y.shape()[1], 1);

    // Salida esperada tras la matemática:
    // L1: [4.1, 6.2, -1.7]
    // ReLU elimina los negativos: [4.1, 6.2, 0.0]
    // L2 suma todo: 4.1 + 6.2 + 0.0 = 10.3
    EXPECT_FLOAT_EQ(Y.data()[0], 10.3f);
}

TEST(SequentialGraphTest, EndToEndWeightInjection) {
    const std::string test_bin = "end_to_end_test.bin";

    // 1. SIMULAR ARCHIVO EXPORTADO DESDE PYTHON (SBS Standard)
    // Capa llamada "hidden" (Linear: 2 in -> 2 out)
    {
        std::ofstream out(test_bin, std::ios::binary);
        out.write("INF1", 4);
        
        uint32_t num_tensors = 2; // weight y bias
        out.write(reinterpret_cast<const char*>(&num_tensors), 4);
        
        // --- hidden.weight (Shape: [2, 2]) ---
        std::string w_name = "hidden.weight";
        uint32_t w_len = w_name.length();
        out.write(reinterpret_cast<const char*>(&w_len), 4);
        out.write(w_name.c_str(), w_len);
        
        uint32_t rank = 2;
        out.write(reinterpret_cast<const char*>(&rank), 4);
        uint32_t dim = 2;
        out.write(reinterpret_cast<const char*>(&dim), 4);
        out.write(reinterpret_cast<const char*>(&dim), 4);
        
        // Matriz de pesos: [[2.0, -1.0], [0.5, 1.0]]
        float w_data[4] = {2.0f, -1.0f, 0.5f, 1.0f};
        out.write(reinterpret_cast<const char*>(w_data), sizeof(w_data));

        // --- hidden.bias (Shape: [2]) ---
        std::string b_name = "hidden.bias";
        uint32_t b_len = b_name.length();
        out.write(reinterpret_cast<const char*>(&b_len), 4);
        out.write(b_name.c_str(), b_len);
        
        uint32_t b_rank = 1;
        out.write(reinterpret_cast<const char*>(&b_rank), 4);
        out.write(reinterpret_cast<const char*>(&dim), 4);
        
        // Sesgos: [0.5, -0.2]
        float b_data[2] = {0.5f, -0.2f};
        out.write(reinterpret_cast<const char*>(b_data), sizeof(b_data));
    }

    // 2. CARGAR EL MODELO DESDE DISCO
    auto state_dict = io::ModelLoader::load<float>(test_bin);

    // 3. CONSTRUIR EL GRAFO Y CARGAR PESOS
    Sequential<float> model;
    // Le pasamos explícitamente el prefijo "hidden" para que coincida con el archivo
    model.add_layer(std::make_unique<Linear<float>>(2, 2), "hidden"); 
    model.add_layer(std::make_unique<Relu<float>>()); // No tiene estado, ignorará los pesos

    // ¡LA LÍNEA MÁGICA DEL PUENTE!
    model.load_state_dict(state_dict);

    // 4. EJECUTAR INFERENCIA (Entrada: [1.0, 3.0])
    Tensor<float> X({1, 2});
    X.data()[0] = 1.0f; X.data()[1] = 3.0f;

    Tensor<float> Y = model.forward(X);

    // 5. VALIDAR RESULTADOS
    // Neurona 0: (1.0 * 2.0) + (3.0 * -1.0) + 0.5 = -0.5 -> ReLU -> 0.0
    // Neurona 1: (1.0 * 0.5) + (3.0 * 1.0) - 0.2 = 3.3  -> ReLU -> 3.3
    ASSERT_EQ(Y.size(), 2);
    EXPECT_FLOAT_EQ(Y.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(Y.data()[1], 3.3f);

    // Limpieza
    std::remove(test_bin.c_str());
}