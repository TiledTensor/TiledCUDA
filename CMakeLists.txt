cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(tiledcuda LANGUAGES C CXX CUDA)

# Prohibit in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source build are not supported")
endif()

option(WITH_TESTING "Build with CTests" ON)
if(WITH_TESTING)
  enable_testing()
endif()

find_package(CUDAToolkit QUIET REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA on)

find_package(Torch REQUIRED)

include(scripts/cmake/generic.cmake)
include(scripts/cmake/dependencies.cmake)

include_directories(include)
add_subdirectory(src)
add_subdirectory(tests/cpp)
