#pragma once

#include "cuda_utils.hpp"
#include "types/common.hpp"

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

/**
 * @brief (haruhi) a naive implementation to experiment with the interface.
 * Consider better implementations.
 *
 * @tparam shm_ptrs: the shared memory positions access by the current threads.
 * @tparam dst: the dst data tile.
 */
template <typename SrcPtrs, typename DstTile>
DEVICE void copy_2d_tile_s2r(const SrcPtrs& shm_ptrs, DstTile& dst) {
    // Not implemented.
}

DEVICE void copy_2d_tile_r2s() {}

}  // namespace tiledcuda::cell::copy
