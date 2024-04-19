#pragma once

#include "cuda_utils.hpp"

namespace tiledcuda::cell::copy {

// Copy a tensor from global memory to shared memory
template <typename Element, typename SrcLayout, typename DstLayout,
          typename TiledCopy>
__forceinline__ __device__ void copy_tensor_g2s(const Element* src_data,
                                                Element* dst_data,
                                                SrcLayout src_layout,
                                                DstLayout dst_layout,
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
__forceinline__ __device__ void copy_tensor_s2g(const Element* src_data,
                                                Element* dst_data,
                                                SrcLayout src_layout,
                                                DstLayout dst_layout,
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

__forceinline__ __device__ void copy_tensor_s2r() {}

__forceinline__ __device__ void copy_tensor_r2s() {}
}  // namespace tiledcuda::cell::copy