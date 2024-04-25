cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(tiledcuda LANGUAGES C CXX CUDA)

# Prohibit in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source build are not supported")
endif()

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

include(cmake/generic.cmake)
include(cmake/dependencies.cmake)

include_directories(include)
add_subdirectory(src)
add_subdirectory(tests/cpp)
