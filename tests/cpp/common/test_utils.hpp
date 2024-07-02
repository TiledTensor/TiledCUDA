#pragma once

#include "config.hpp"

#include <cutlass/half.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace tiledcuda::testing {

void assert_equal(const __half* v1, const __half* v2, int64_t numel);

}  // namespace tiledcuda::testing
