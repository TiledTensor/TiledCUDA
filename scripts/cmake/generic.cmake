set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "The C++ standard whoese features are requested." FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD
    17
    CACHE STRING "The CUDA standard whose features are requested." FORCE)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set host compiler flags. Enable all warnings and treat them as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

# let cmake set PYTHON_EXECUTABLE to suppress some warnings when building with
# torchlibs
find_package(PythonInterp 3 REQUIRED)
message(STATUS "Python interpreter path: ${PYTHON_EXECUTABLE}")

# set device compiler flags
cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ARCH_FLAGS} --Werror all-warnings")

# FIXME(haruhi): -std=c++17 has to be set explicitly here, otherwise compiling
# with torchlibs raises errors
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++17 ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -std=c++17 -O0 ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} --std=c++17 -O3
                            ${ARCH_FLAGS})

# Set the CUDA_PROPAGATE_HOST_FLAGS to OFF to avoid passing host compiler flags
# to the device compiler
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

message(STATUS "TiledCUDA: CUDA detected: " ${CUDA_VERSION})
message(STATUS "TiledCUDA: CUDA nvcc is: " ${CUDA_NVCC_EXECUTABLE})
message(STATUS "TiledCUDA: CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})
message(STATUS "CUDA Architecture flags = ${ARCH_FLAGS}")

function(cuda_test TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(nv_test "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  cuda_add_executable(
    ${TARGET_NAME} ${PROJECT_SOURCE_DIR}/tests/cpp/test_unit.cc ${nv_test_SRCS})
  target_link_libraries(${TARGET_NAME} ${nv_test_DEPS} gtest glog::glog)
  add_dependencies(${TARGET_NAME} ${nv_test_DEPS} gtest glog::glog)

  # add a test with the same name as the target
  add_test(${TARGET_NAME} ${TARGET_NAME})
endfunction(cuda_test)
