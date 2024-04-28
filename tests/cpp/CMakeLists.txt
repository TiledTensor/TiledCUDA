set(TEST_ROOT ${PROJECT_SOURCE_DIR}/tests/cpp)
message(STATUS "TEST_ROOT is set to: ${TEST_ROOT}")

include_directories(${TEST_ROOT})

if(WITH_TESTING)
  cuda_test(test_copy SRCS ${TEST_ROOT}/cell/test_copy.cu)
  cuda_test(test_gemm SRCS ${TEST_ROOT}/kernels/test_gemm.cu)
  cuda_test(test_scatter_nd SRCS ${TEST_ROOT}/kernels/test_scatter_nd.cu)
endif()
