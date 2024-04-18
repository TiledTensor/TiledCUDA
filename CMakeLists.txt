cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDACXX /usr/local/cuda/bin/nvcc)

project(tiledcuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-dc;-dlink")
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# set(CUDA_SEPARABLE_COMPILATION ON)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
# find_package(CuDNN REQUIRED)


file(GLOB_RECURSE SOURCES 
    "src/kernels/*.cu" 
    "src/*.cc"
)

# Define our library target
add_library(tiledcuda SHARED ${SOURCES})

target_compile_options(tiledcuda PRIVATE -std=c++17)
target_compile_options(tiledcuda PRIVATE -fconcepts)
target_compile_options(tiledcuda PRIVATE -fpermissive)

include_directories(include)

include_directories(3rd-party/cutlass/include)


# Link against LibTorch
target_link_libraries(tiledcuda "${TORCH_LIBRARIES}")
target_link_libraries(tiledcuda "${CUDA_LIBRARIES}")
target_link_libraries(tiledcuda CUDA::cudart)

# target_link_libraries(tiledcuda "${CUDNN_LIBRARY}")

# set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
# set_property(TARGET tiledcuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)