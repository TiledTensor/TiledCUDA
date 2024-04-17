#pragma once

#include "cuda_utils.hpp"

namespace tiledcuda::cell::copy {

// Copy a tensor from global memory to shared memory
template <typename Element, typename SrcLayout, typename DstLayout,
          typename TiledCopy>
__device__ void copy_tensor_g2s(const Element* src_data, Element* dst_data,
                                SrcLayout src_layout, DstLayout dst_layout,
                                TiledCopy tiled_copy, int tid);

// Copy a tensor from shared memory to global memory
template <typename Element, typename SrcLayout, typename DstLayout,
          typename TiledCopy>
__device__ void copy_tensor_s2g(const Element* src_data, Element* dst_data,
                                SrcLayout src_layout, DstLayout dst_layout,
                                TiledCopy tiled_copy, int tid);
}  // namespace tiledcuda::cell