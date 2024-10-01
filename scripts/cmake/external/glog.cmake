include(ExternalProject)

set(GLOG_REPOSITORY https://github.com/google/glog.git)
set(GLOG_TAG v0.7.1)

set(GLOG_PREFIX_DIR ${THIRD_PARTY_BUILD_DIR}/glog)
set(GLOG_SOURCE_DIRS ${GLOG_PREFIX_DIR}/src/extern_glog)
set(GLOG_INSTALL_DIR ${THIRD_PARTY_BUILD_DIR}/install/glog)

set(GLOG_INCLUDE_DIRS
    "${GLOG_INSTALL_DIR}/include"
    CACHE PATH "glog include directory." FORCE)
set(GLOG_LIBRARIES
    "${GLOG_INSTALL_DIR}/lib/libglog.so"
    CACHE FILEPATH "glog library." FORCE)

ExternalProject_Add(
  extern_glog
  GIT_REPOSITORY ${GLOG_REPOSITORY}
  GIT_TAG ${GLOG_TAG}
  GIT_SHALLOW TRUE
  PREFIX ${GLOG_PREFIX_DIR}
  SOURCE_DIR ${GLOG_SOURCE_DIR}
  INSTALL_DIR ${GLOG_INSTALL_DIR}
  BUILD_IN_SOURCE 1
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GLOG_INSTALL_DIR}
             -DCMAKE_INSTALL_LIBDIR=${GLOG_INSTALL_DIR}/lib
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DWITH_GFLAGS=OFF
             -DWITH_GTEST=OFF
             -DBUILD_TESTING=OFF)

add_library(glog::glog INTERFACE IMPORTED)
include_directories(glog::glog BEFORE SYSTEM ${GLOG_INCLUDE_DIRS})
target_link_libraries(glog::glog INTERFACE ${GLOG_LIBRARIES})
add_dependencies(glog::glog extern_glog)
