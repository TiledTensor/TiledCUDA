set(TARGET "tiledcuda")

file(GLOB_RECURSE SOURCES "kernels/*.cu" "*.cc")

# Define our library target
cuda_add_library(${TARGET} SHARED ${SOURCES})

target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -std=c++17
                                         -fconcepts -fpermissive>)

# for linking glog, in the call of `target_link_libraries` below, the header
# files is included directly, there is no need to explicitly call
# `target_include_directories`. Link against LibTorch.
target_link_libraries(${TARGET} "${TORCH_LIBRARIES}" "${CUDA_LIBRARIES}"
                      CUDA::cudart glog::glog)
