cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tiledcuda LANGUAGES C CXX CUDA)

# Prohibit in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source build are not supported")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${PROJECT_SOURCE_DIR}/scripts/cmake")

option(WITH_TESTING "Build with CTests" ON)
if(WITH_TESTING)
  enable_testing()
endif()

option(ENABLE_DEBUG "Enable debug mode" ON)

include(generic)
include(dependencies)


include_directories(include)
add_subdirectory(src)
add_subdirectory(tests/cpp)

# TODO(KuangjuX): Traverse and query the examples under the `examples` directory, and include them.
add_subdirectory(examples/cpp/gemm)
add_subdirectory(examples/cpp/b2b_gemm)
add_subdirectory(examples/cpp/flash_attn)
