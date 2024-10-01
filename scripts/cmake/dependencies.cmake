# set the third party directory for dependencies that do not need a build
set(THIRD_PARTY_DIR
    "${PROJECT_SOURCE_DIR}/3rd-party"
    CACHE STRING
          "A path that specifies the directory for third-party downloads.")

# set the third party build directory for dependencies that need a build
set(THIRD_PARTY_BUILD_DIR
    "${CMAKE_BINARY_DIR}/3rd-party"
    CACHE STRING
          "A path that specifies the directory for third-party build & install."
)

set(THIRD_PARTY_BUILD_TYPE Release)

# add cutlass into dependence
include_directories(${THIRD_PARTY_DIR}/cutlass/include)

# add googletest into dependence
set(INSTALL_GTEST
    OFF
    CACHE BOOL "Install gtest." FORCE)
add_subdirectory(${THIRD_PARTY_DIR}/googletest)
include_directories(BEFORE SYSTEM
                    ${THIRD_PARTY_DIR}/googletest/googletest/include)

# add glog into dependence
include(${PROJECT_SOURCE_DIR}/scripts/cmake/public/glog.cmake)
