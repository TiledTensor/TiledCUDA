cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gemm_examples LANGUAGES C CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${PROJECT_SOURCE_DIR}/../../../scripts/cmake")
include(generic)

set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/../../../3rd-party")
include_directories("${THIRD_PARTY_DIR}/cutlass/include")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/../../../include")

add_executable(01_gemm_global_reg 01_gemm_global_reg/main.cu)
target_link_libraries(01_gemm_global_reg ${CUDA_CUBLAS_LIBRARIES})

add_executable(02_gemm_all_mem 02_gemm_all_mem/main.cu)
target_link_libraries(02_gemm_all_mem ${CUDA_CUBLAS_LIBRARIES})
