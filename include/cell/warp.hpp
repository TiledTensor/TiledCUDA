#pragma once

#include "cuda_utils.hpp"

namespace tiledcuda::cell {

/**
 * @brief `shuffle_sync` provides a way of moving a value from one thread
 * to other threads in the warp in one instruction.
 */
template <typename Element>
DEVICE Element shuffle_sync(uint32_t mask, Element value, int src_lane) {
    return __shfl_sync(mask, value, src_lane);
}

/**
 * @brief `shuffle_down_sync` enable you to shift data within a warp in
 * one instruction. So basically the `value` is shiffted down `delta`
 * lanes.
 */
template <typename Element>
DEVICE Element shuffle_down_sync(uint32_t mask, Element value, int delta) {
    return __shfl_down_sync(mask, value, delta);
}

}  // namespace tiledcuda::cell
