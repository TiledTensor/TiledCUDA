#pragma once

#include "cuda_utils.hpp"

namespace tiledcuda::cell {

template <typename Element>
DEVICE Element shuffle_down_sync(uint32_t mask, Element value, int delta) {
    return __shfl_down_sync(mask, value, delta);
}

}  // namespace tiledcuda::cell
