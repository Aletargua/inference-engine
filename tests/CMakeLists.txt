# =============================================================================
# @file tests/CMakeLists.txt
# @brief Build configuration for GoogleTest unit tests (TDD).
# =============================================================================

# Create the test executables.
add_executable(test_tensor test_tensor.cpp)
add_executable(test_ops_linear test_ops_linear.cpp)
add_executable(test_ops_relu test_ops_relu.cpp)
add_executable(test_graph_sequential test_graph_sequential.cpp)

# Link the test binaries against our engine (infer_core) and GTest.
# PRIVATE: The test needs these libraries, but it doesn't propagate them further.
target_link_libraries(test_tensor 
    PRIVATE 
        infer_core 
        GTest::gtest_main
)

target_link_libraries(test_ops_linear 
    PRIVATE 
        infer_core 
        GTest::gtest_main
)

target_link_libraries(test_ops_relu              
    PRIVATE 
        infer_core 
        GTest::gtest_main
)

target_link_libraries(test_graph_sequential 
    PRIVATE 
        infer_core 
        GTest::gtest_main
)

# Instruct CMake to discover and register GTest macros (TEST, TEST_F) automatically.
include(GoogleTest)
gtest_discover_tests(test_tensor)
gtest_discover_tests(test_ops_linear)
gtest_discover_tests(test_ops_relu)
gtest_discover_tests(test_graph_sequential)       








