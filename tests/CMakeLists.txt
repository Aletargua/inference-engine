# =============================================================================
# @file tests/CMakeLists.txt
# @brief Build configuration for GoogleTest unit tests (TDD).
# =============================================================================

# Create the test executable.
add_executable(test_tensor test_tensor.cpp)
add_executable(test_ops_linear test_ops_linear.cpp)

# Link the test binary against our engine (infer_core) and GTest.
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

# Instruct CMake to discover and register GTest macros (TEST, TEST_F) automatically.
include(GoogleTest)
gtest_discover_tests(test_tensor)
gtest_discover_tests(test_ops_linear)