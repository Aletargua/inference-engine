# =============================================================================
# @file src/CMakeLists.txt
# @brief Build configuration for the core Inference Engine binaries.
# =============================================================================

# Define the core static/shared library for the engine.
add_library(infer_core 
    core/tensor.cpp
    ops/linear.cpp
    ops/relu.cpp
    graph/sequential.cpp
    io/model_loader.cpp
)
# Link the dependencies.
# PUBLIC: Anyone linking against 'infer_core' will also inherit 'infer_headers' and 'TBB'.
target_link_libraries(infer_core 
    PUBLIC 
        infer_headers 
        TBB::tbb
)