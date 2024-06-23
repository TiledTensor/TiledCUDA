#pragma once

#include "cuda_utils.hpp"

#include <cute/tensor.hpp>

namespace tiledcuda::cell::copy {

using namespace cute;

template <typename TiledCopy, typename STensor, typename DTensor,
          typename DTensorView>
struct Shm2RegLoad {
  public:
    DEVICE Shm2RegLoad(TiledCopy& copy, const STensor& src, DTensor& dst,
                       DTensorView& dst_view)
        : tiled_copy_(copy), src_(src), dst_(dst), dst_view_(dst_view) {}

    DEVICE void copy(int pos) {
        cute::copy(tiled_copy_, src_(_, _, pos), dst_view_(_, _, pos));
    }

    DEVICE int get_iters() { return size<2>(dst_); }

    DEVICE const auto operator[](int idx) { return dst_(_, _, idx); }

  private:
    TiledCopy& tiled_copy_;
    const STensor& src_;
    DTensor& dst_;
    DTensorView& dst_view_;
};

template <typename Element, typename Layout, typename TiledMma>
DEVICE auto make_s2rA(const Element* data, const Layout& layout,
                      const TiledMma& tiled_mma) {
    int tid = threadIdx.x;

    auto tensor = cute::make_tensor(make_smem_ptr(data), layout);

    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    auto tiled_copy = make_tiled_copy_A(SmemLoadAtom{}, tiled_mma);

    auto thrd_copy = tiled_copy.get_thread_slice(tid);
    auto src = thrd_copy.partition_S(tensor);

    // partition register
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto dst = thr_mma.partition_fragment_A(tensor);
    auto dst_view = thrd_copy.retile_D(dst);

    Shm2RegLoad loader(tiled_copy, src, dst, dst_view);
    return loader;
}

// FIXIME(haruhi): the current implementation is for fast experiment,
// it is coupled shared memory layout with the register layout
template <typename Element, typename Layout, typename TiledMma>
DEVICE auto make_s2rB(const Element* data, const Layout& layout,
                      const TiledMma& tiled_mma) {
    int tid = threadIdx.x;

    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    auto tiled_copy = make_tiled_copy_B(SmemLoadAtom{}, tiled_mma);
    auto thrd_copy = tiled_copy.get_thread_slice(tid);

    auto tensor = make_tensor(make_smem_ptr(data), layout);
    auto src = thrd_copy.partition_S(tensor);

    // partition register
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto dst = thr_mma.partition_fragment_B(tensor);
    auto dst_view = thrd_copy.retile_D(dst);

    Shm2RegLoad loader(tiled_copy, src, dst, dst_view);
    return loader;
}
}  // namespace tiledcuda::cell::copy
