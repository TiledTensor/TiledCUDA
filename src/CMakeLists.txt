set(TARGET "tiledcuda")

file(GLOB_RECURSE SOURCES "kernels/*.cu" "*.cc")

# Define our library target
cuda_add_library(${TARGET} SHARED ${SOURCES})

set_target_properties(
  ${TARGET}
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             CUDA_EXTENSIONS OFF
             CUDA_RESOLVE_DEVICE_SYMBOLS ON
             CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(
  ${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: -Werror,-Wall -rdc=true
                   -std=c++20 -fconcepts -fpermissive>)
target_compile_features(${TARGET} PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(${TARGET} "${TORCH_LIBRARIES}")
