#pragma once

#include "cuda_utils.hpp"

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace tiledcuda::cell::copy {
using namespace cute;

template <typename Element, typename TiledMma_, typename DstLayout>
struct R2SCopy2D {
    using TiledMma = TiledMma_;
    using Dstlayout_ = DstLayout;
    using CopyAtom = Copy_Atom<DefaultCopy, Element>;

   public:
    template <typename Engine, typename Layout>
    DEVICE void copy(cute::Tensor<Engine, Layout> const& acc, Element* dst_data,
                     int tid) {
        // FIXME(haruhi): This implementation is specifically designed
        // for tcu WMMA and assumes that the ACC value has a
        // floating-point precision. The code converts the ACC value
        // to half-precision.
        auto src_tensor = convert_type<Element>(acc);
        auto dst_tensor = make_tensor(make_smem_ptr(dst_data), DstLayout{});

        auto tiled_copy = make_tiled_copy_C(CopyAtom{}, TiledMma{});
        auto thrd_copy = tiled_copy.get_thread_slice(tid);

        auto src = thrd_copy.retile_S(src_tensor);
        auto dst = thrd_copy.partition_D(dst_tensor);
        cute::copy(tiled_copy, src, dst);
    }

   private:
    template <typename To_type, typename Engine, typename Layout>
    DEVICE auto convert_type(cute::Tensor<Engine, Layout> const& tensor) {
        using From_type = typename Engine::value_type;
        constexpr int numel = decltype(size(tensor))::value;
        cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
        // HACK: this requires tensor to be "contiguous"
        auto frag = convert_op(
            *reinterpret_cast<const cutlass::Array<From_type, numel>*>(
                tensor.data()));
        return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    }
};

namespace {
template <typename Element, typename Layout, typename TiledMma>
DEVICE auto make_s2rA(const Element* data, int tid, const Layout& layout,
                      const TiledMma& tiled_mma) {
    auto tensor = make_tensor(make_smem_ptr(data), layout);

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
}  // namespace

// @tparam src_ptr: pointer to the source data
// @tparam src_layout: layout of the source data
// @tparam dst_ptr: pointer to the destination data
// @tparam dst_layout: layout of the destination data
// @tparam thread_layout: layout of the thread
// @tparam tid: thread id
template <typename Element, typename SrcLayout, typename DstLayout,
          typename ThreadLayout>
DEVICE void copy_2d_tile_s2r(const Element* src, Element* dst, int tid) {
    auto tensor = make_tensor(make_smem_ptr(src), SrcLayout{});

    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    auto tiled_copy = make_tiled_copy_A(SmemLoadAtom{}, tiled_mma);

    using GmemTiledCopy =
        decltype(make_tiled_copy(SmemLoadAtom{}, ThreadLayout{}, ValLayout{}));

    // auto thrd_copy = tiled_copy.get_thread_slice(tid);
    // auto src = thrd_copy.partition_S(tensor);
}

DEVICE void copy_2d_tile_r2s() {}

}  // namespace tiledcuda::cell::copy
