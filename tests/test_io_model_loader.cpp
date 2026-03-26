    // ============================================================================
// @file tests/test_io_model_loader.cpp
// @brief Unit tests for the ModelLoader and SBS binary parsing.
// ============================================================================

#include <gtest/gtest.h>
#include "io/model_loader.hpp"
#include <fstream>
#include <cstdio> // For std::remove

using namespace infer;
using namespace infer::io;

TEST(ModelLoaderTest, LoadValidSBSFile) {
    const std::string test_filename = "test_dummy_weights.bin";

    // 1. CREATE A DUMMY BINARY FILE (Simulating the Python Export)
    {
        std::ofstream out(test_filename, std::ios::binary);
        out.write("INF1", 4);
        
        uint32_t num_tensors = 1;
        out.write(reinterpret_cast<const char*>(&num_tensors), 4);
        
        std::string name = "hidden.weight";
        uint32_t name_len = name.length();
        out.write(reinterpret_cast<const char*>(&name_len), 4);
        out.write(name.c_str(), name_len);
        
        uint32_t rank = 2;
        out.write(reinterpret_cast<const char*>(&rank), 4);
        
        uint32_t dim0 = 2, dim1 = 3;
        out.write(reinterpret_cast<const char*>(&dim0), 4);
        out.write(reinterpret_cast<const char*>(&dim1), 4);
        
        float data[6] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f};
        out.write(reinterpret_cast<const char*>(data), sizeof(data));
    }

    // 2. TEST THE C++ LOADER
    auto loaded_tensors = ModelLoader::load<float>(test_filename);

    // 3. VALIDATE RESULTS
    ASSERT_EQ(loaded_tensors.size(), 1);
    ASSERT_TRUE(loaded_tensors.find("hidden.weight") != loaded_tensors.end());

    const auto& tensor = loaded_tensors.at("hidden.weight");
    EXPECT_EQ(tensor.shape().size(), 2);
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 3);
    EXPECT_FLOAT_EQ(tensor.data()[0], 1.1f);
    EXPECT_FLOAT_EQ(tensor.data()[5], 6.6f);

    // 4. CLEANUP (Delete test file)
    std::remove(test_filename.c_str());
}

TEST(ModelLoaderTest, ThrowsOnInvalidMagicNumber) {
    const std::string test_filename = "test_bad_magic.bin";
    {
        std::ofstream out(test_filename, std::ios::binary);
        out.write("BAD0", 4); // Invalid magic number
    }

    EXPECT_THROW({
        ModelLoader::load<float>(test_filename);
    }, std::runtime_error);

    std::remove(test_filename.c_str());
}