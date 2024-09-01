# CMake 3.25 is required for CUDA 20.
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
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

option(ENABLE_DEBUG "Enable debug mode" OFF)

include(generic)
include(dependencies)


include_directories(include)
add_subdirectory(src)
add_subdirectory(tests/cpp)


set(EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp")
file(GLOB_RECURSE SUBDIRS "${EXAMPLES_DIR}/*")

foreach(EXAMPLE_DIR ${SUBDIRS})
  if(IS_DIRECTORY ${EXAMPLE_DIR})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_DIR} NAME)
    add_subdirectory(${EXAMPLE_DIR})
  endif()
endforeach()
