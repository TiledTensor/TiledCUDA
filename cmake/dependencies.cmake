# add cutlass into dependence
include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rd-party/cutlass/include)

# add googletest into dependence
set(INSTALL_GTEST
    OFF
    CACHE BOOL "Install gtest." FORCE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../3rd-party/googletest)
include_directories(
  BEFORE SYSTEM
  ${CMAKE_CURRENT_LIST_DIR}/../3rd-party/googletest/googletest/include)
