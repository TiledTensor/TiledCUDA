set(THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/../../3rd-party")

# add cutlass into dependence
include_directories("${THIRD_PARTY_DIR}/cutlass/include")

# add googletest into dependence
set(INSTALL_GTEST
    OFF
    CACHE BOOL "Install gtest." FORCE)
add_subdirectory("${THIRD_PARTY_DIR}/googletest")
include_directories(BEFORE SYSTEM
                    "${THIRD_PARTY_DIR}/googletest/googletest/include")
