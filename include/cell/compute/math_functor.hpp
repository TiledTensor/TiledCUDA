#pragma once

#include "cuda_utils.hpp"

namespace tiledcuda::cell::compute {

template <typename Element>
struct Add {
    DEVICE Element operator()(Element a, Element b) const { return a + b; }
};

template <typename Element>
struct Mul {
    DEVICE Element operator()(Element a, Element b) const { return a * b; }
};

template <typename Element>
struct Max {
    DEVICE Element operator()(Element a, Element b) const {
        return a > b ? a : b;
    }
};

template <typename Element>
struct Min {
    DEVICE Element operator()(Element a, Element b) const {
        return a < b ? a : b;
    }
};

}  // namespace tiledcuda::cell::compute
