set(GLOG_FIND_REQUIRED ON)

# find locally installed glog
find_package(glog CONFIG QUIET) # try to use the config mode first
if(NOT TARGET glog::glog)
  find_package(glog MODULE QUIET)
endif()

if(TARGET glog::glog)
  set(GLOG_FOUND TRUE)
  message(STATUS "Found Glog at ${glog_DIR}")
  message(STATUS "  Target : glog::glog")
else()
  message(
    STATUS "Could not find Glog automatically with new-style glog target, "
           "use legacy find.")
  # Try to find glog manually. Older versions of glog do not include a
  # find_package configuration. Therefore, we must use custom logic to locate
  # the library and remap it to an imported target.

  include(FindPackageHandleStandardArgs)

  # set path search hints
  list(APPEND GLOG_CHECK_INCLUDE_DIRS /usr/local/include /opt/local/include
       /usr/include)
  list(APPEND GLOG_CHECK_PATH_SUFFIXES glog/include glog/Include Glog/include
       Glog/Include)

  list(APPEND GLOG_CHECK_LIBRARY_DIRS /usr/local/lib /opt/local/lib /usr/lib)
  list(
    APPEND
    GLOG_CHECK_LIBRARY_SUFFIXES
    glog/lib
    glog/Lib
    glog/lib64
    Glog/lib
    Glog/Lib
    x64/Release)

  find_path(
    GLOG_INCLUDE_DIRS
    NAMES glog/logging.h
    PATHS ${GLOG_INCLUDE_DIR_HINTS} ${GLOG_CHECK_INCLUDE_DIRS}
    PATH_SUFFIXES ${GLOG_CHECK_PATH_SUFFIXES})

  find_library(
    GLOG_LIBRARIES
    NAMES glog libglog
    PATHS ${GLOG_LIBRARY_DIR_HINTS} ${GLOG_CHECK_LIBRARY_DIRS}
    PATH_SUFFIXES ${GLOG_CHECK_LIBRARY_SUFFIXES})

  if(GLOG_INCLUDE_DIRS AND GLOG_LIBRARIES)
    set(GLOG_FOUND TRUE)
    message(STATUS "Found Glog")
    message(STATUS "  Includes : ${GLOG_INCLUDE_DIRS}")
    message(STATUS "  Libraries : ${GLOG_LIBRARIES}")

    add_library(glog::glog INTERFACE IMPORTED)
    target_include_directories(glog::glog INTERFACE ${GLOG_INCLUDE_DIRS})
    target_link_libraries(glog::glog INTERFACE ${GLOG_LIBRARIES})
  endif()
endif()

if(NOT GLOG_FOUND AND GLOG_FIND_REQUIRED)
  # If glog is not installed locally, download and build it using
  # ExternalProject.
  include(external/glog)
endif()
