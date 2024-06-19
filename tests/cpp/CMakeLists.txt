if(WITH_TESTING)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  file(
    GLOB_RECURSE UNIT_TESTS
    LIST_DIRECTORIES FALSE
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.cu")

  foreach(FILE_PATH ${UNIT_TESTS})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_NAME ${FILE_PATH})
    string(REPLACE ".cu" "" TEST_NAME "${FILE_NAME}")

    cuda_test(${TEST_NAME} SRCS "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_PATH}")
  endforeach()
endif()
