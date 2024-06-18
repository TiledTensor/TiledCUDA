#pragma once

#include "cuda_utils.hpp"
#include "types/common.hpp"
#include "types/tile_shape.hpp"

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
    DEVICE void copy(cute::Tensor<Engine, Layout> const& acc,
                     Element* dst_data) {
        int tid = threadIdx.x;

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

namespace detail {

enum class CopyInst {
    Ldmatrix = 0,  // ldmatrix for loading data from shared memory to register.
    Stmatrix = 1,
    Ldsm32 = 2,
    Ldsm128 = 3
};

// functor to copy data from shared memory to register file.
template <typename Shared, typename Reg, CopyInst kCopyInst>
struct CopyShared2Reg {
    DEVICE void operator()();
};

// partial specialization for ldmatrix
template <typename Shared, typename Reg>
struct CopyShared2Reg<Shared, Reg, CopyInst::Ldmatrix> {
    DEVICE void operator()(const Shared& src, Reg& dst) {}
};

// functor to copy data from shared memory to register file.
template <typename Reg, typename Shared, typename InstShape, CopyInst kCopyInst>
struct CopyReg2Shared {
    DEVICE void operator()();
};

// partial specialization for wmma 16x16x16, and LDSM32
template <typename Reg, typename Shared>
struct CopyReg2Shared<Reg, Shared, InstShape<16, 16, 16>, CopyInst::Ldsm32> {
    DEVICE void operator()(const Reg& src, Shared& dst) {}
};

}  // namespace detail

/// @brief Copy a tile from shared memory: to register.
/// @tparam Shared the shared memory tile type.
/// @tparam Reg the register tile type.
/// @tparam WarpLayout the warp layout.
template <typename Shared, typename Reg, typename WarpLayout>
DEVICE void copy_tile_s2r(const Shared& src, Reg& dst,
                          const WarpLayout& layout /*for auto type-infer*/) {
    using Copy =
        detail::CopyShared2Reg<Shared, Reg, detail::CopyInst::Ldmatrix>;

    Copy copy;
    copy(src, dst);
}

template <typename Reg, typename Shared, typename WarpLayout>
DEVICE void copy_tile_r2s(const Reg& src, Shared& dst,
                          const WarpLayout& layout /*for auto type infer*/) {
    using Copy = detail::CopyReg2Shared<Reg, Shared, InstShape<16, 16, 16>,
                                        detail::CopyInst::Ldsm32>;
    Copy copy;
    copy(src, dst);
}

}  // namespace tiledcuda::cell::copy
