set(TARGET "tiledcuda")

file(GLOB_RECURSE SOURCES "kernels/*.cu" "*.cc")

# Define our library target
cuda_add_library(${TARGET} SHARED ${SOURCES})

target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -std=c++17
                                         -fconcepts -fpermissive>)

# Link against LibTorch
target_link_libraries(${TARGET} "${TORCH_LIBRARIES}" "${CUDA_LIBRARIES}"
                      CUDA::cudart)
