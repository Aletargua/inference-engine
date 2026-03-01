// ============================================================================
// @file tests/test_graph_sequential.cpp
// @brief Integration test for the Sequential orchestrator (Mini-MLP).
// ============================================================================

#include <gtest/gtest.h>
#include "core/tensor.hpp"
#include "ops/linear.hpp"
#include "ops/relu.hpp"
#include "graph/sequential.hpp"

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