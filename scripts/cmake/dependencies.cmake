set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/3rd-party")

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
