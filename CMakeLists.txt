cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Prohibit in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source build are not supported")
endif()

project(tiledcuda LANGUAGES C CXX CUDA)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "The C++ standard whoese features are requested.")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(cmake/Dependencies.cmake)

# Set host compiler flags. Enable all warnings and treat them as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

find_package(Torch REQUIRED)

find_package(CUDA REQUIRED)
cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS}  ${ARCH_FLAGS}  --Werror all-warnings")
# Set the CUDA_PROPAGATE_HOST_FLAGS to OFF to avoid passing host compiler flags
# to the device compiler
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

message(STATUS "TiledCUDA: CUDA detected: " ${CUDA_VERSION})
message(STATUS "TiledCUDA: CUDA nvcc is: " ${CUDA_NVCC_EXECUTABLE})
message(STATUS "TiledCUDA: CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})
message(STATUS "CUDA Architecture flags = ${ARCH_FLAGS}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -w ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -w ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -w -O3 ${ARCH_FLAGS})

file(GLOB_RECURSE SOURCES "src/kernels/*.cu" "src/*.cc")

# Define our library target
cuda_add_library(tiledcuda SHARED ${SOURCES})

target_compile_options(tiledcuda PRIVATE -std=c++17)
target_compile_options(tiledcuda PRIVATE -fconcepts)
target_compile_options(tiledcuda PRIVATE -fpermissive)

include_directories(include)

# Link against LibTorch
target_link_libraries(tiledcuda "${TORCH_LIBRARIES}")
target_link_libraries(tiledcuda "${CUDA_LIBRARIES}")
target_link_libraries(tiledcuda CUDA::cudart)
