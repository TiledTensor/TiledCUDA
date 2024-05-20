#pragma once

#include "cuda_utils.hpp"

#include <cute/tensor.hpp>

namespace tiledcuda::cell::copy {

using namespace cute;

// Copy a 2d data tile from global memory to shared memory
template <typename Element, typename SrcLayout, typename DstLayout,
          typename TiledCopy>
DEVICE void copy_2d_tile_g2s(const Element* src_data, Element* dst_data,
                             SrcLayout src_layout, DstLayout dst_layout,
                             TiledCopy tiled_copy, int tid) {
    auto gtile = make_tensor(make_gmem_ptr(src_data), src_layout);
    auto stile = make_tensor(make_smem_ptr(dst_data), dst_layout);

    auto loader = tiled_copy.get_thread_slice(tid);

    auto src = loader.partition_S(gtile);
    auto dst = loader.partition_D(stile);

#pragma unroll
    for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
        for (int j = 0; j < int(size<2>(src)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst(_, i, j));
}

// Copy a tensor from shared memory to global memory
template <typename Element, typename SrcLayout, typename DstLayout,
          typename TiledCopy>
DEVICE void copy_2d_tile_s2g(const Element* src_data, Element* dst_data,
                             SrcLayout src_layout, DstLayout dst_layout,
                             TiledCopy tiled_copy, int tid) {
    auto stile = make_tensor(make_smem_ptr(src_data), src_layout);
    auto gtile = make_tensor(make_gmem_ptr(dst_data), dst_layout);

    auto loader = tiled_copy.get_thread_slice(tid);

    auto src = loader.partition_S(stile);
    auto dst = loader.partition_D(gtile);

#pragma unroll
    for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
        for (int j = 0; j < int(size<2>(src)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst(_, i, j));
}

}  // namespace tiledcuda::cell::copy
