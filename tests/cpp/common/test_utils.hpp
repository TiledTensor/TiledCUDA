#pragma once

#include "config.hpp"

#include <cutlass/half.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace tiledcuda::testing {

template <typename T>
void assert_equal(const T* v1, const T* v2, int64_t numel, float epsilon);

}  // namespace tiledcuda::testing
