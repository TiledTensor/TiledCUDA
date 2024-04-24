set(SUBDIRS cell kernels)

foreach(SUBDIR ${SUBDIRS})
  add_subdirectory(${SUBDIR})
endforeach()
