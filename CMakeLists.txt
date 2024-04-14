cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDACXX /usr/local/cuda/bin/nvcc)

project(tiledcuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
# find_package(CuDNN REQUIRED)

add_compile_definitions(-DCUTLASS_NVCC_ARCHS=86)

file(GLOB_RECURSE SOURCES "src/kernels/*.cu" "src/kernel/cell/*.cc")

# Define our library target
add_library(tiledcuda SHARED ${SOURCES})

target_compile_options(tiledcuda PRIVATE -std=c++17)

include_directories(include)

include_directories(3rd-party/cutlass/include)

# target_compile_features(tiledcuda PRIVATE cxx_std_17)

# Link against LibTorch
target_link_libraries(tiledcuda "${TORCH_LIBRARIES}")
target_link_libraries(tiledcuda "${CUDA_LIBRARIES}")
# target_link_libraries(tiledcuda "${CUDNN_LIBRARY}")

# set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")