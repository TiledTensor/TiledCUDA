#pragma once

#include "cuda_utils.hpp"

namespace tiledcuda::cell::compute {

template <typename Element>
struct Add {
    DEVICE void operator()(Element a, Element b) const { return a + b; }
};

template <typename Element>
struct Max {
    DEVICE void operator()(Element a, Element b) const { return a > b ? a : b; }
};

}  // namespace tiledcuda::cell::compute