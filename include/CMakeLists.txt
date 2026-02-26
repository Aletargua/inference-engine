# =============================================================================
# @file include/CMakeLists.txt
# @brief Interface configuration for the Inference Engine headers.
# =============================================================================

# Create an INTERFACE library. 
# Interface libraries do not compile to binary objects (.o or .a); 
# they purely propagate usage requirements (like include directories) to dependents.
add_library(infer_headers INTERFACE)

# Expose the current directory so targets linking to 'infer_headers' 
# can do #include "core/tensor.hpp" natively.
target_include_directories(infer_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})