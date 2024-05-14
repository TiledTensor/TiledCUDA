#pragma once

#include "cell/copy/static_copy.hpp"
#include "cell/traits/base.hpp"
#include "types/layout.hpp"
#include "types/tile_shape.hpp"

#include <cute/arch/copy.hpp>
#include <cute/tensor.hpp>

#include <type_traits>

namespace tiledcuda::cell::traits {

using namespace cute;

template <typename Element_, typename CtaTileShape, typename WarpShape,
          typename Base = TraitsBase<Element_>>
struct DynBack2BackGemmTraits : public Base {
    using Element = Element_;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;
    static constexpr int kTP = dim_size<3, CtaTileShape>;

    static_assert(kTK == kTN && kTN == kTP,
                  "B2B GEMM requires kTK == kTN == kTP.");

    static constexpr int kWarpPerRow = dim_size<0, WarpShape>;
    static constexpr int kWarpPerCol = dim_size<1, WarpShape>;
    static_assert(kWarpPerCol == 1,
                  "The B2B GEMM requires a single warp along CTA tile.");

    static constexpr int kThreads = kWarpPerRow * kWarpPerCol * 32;

    static constexpr int kNumPerAccess = Base::kNumPerAccess;
    static constexpr int kThreadsPerCol = CeilDiv<kTK, Base::kNumPerAccess>;
    static constexpr int kThreadsPerRow = CeilDiv<kThreads, kThreadsPerCol>;

    static constexpr int kSwizzle = (kTK == 32 ? 2 : 3);

    using SmemLayoutAtom = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kTK>>, Stride<Int<kTK>, _1>>{}));

    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));

    // The current implementation requires B are laid out in column
    // major. a [kTK, kTN] matrix in column major can be interpreted
    // as a [kTN, kTK] matrix in row major.
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));
    // a [kTN, kTP] matrix in column major fashion,
    // can be interpreted as a [kTP, kTN] matrix in row major
    // fashion.
    using SmemLayoutC =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTP>, Int<kTN>>{}));

    using ThreadShape = TileShape<kThreadsPerRow, kThreadsPerCol>;

    using ThreadLayout = Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
                                Stride<Int<kThreadsPerCol>, _1>>;

    static const bool enable_cp_async = false;  // change this flag
    using CopyInst = std::conditional_t<
        enable_cp_async,
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>,
        Copy_Atom<DefaultCopy, Element>>;

    using TiledCopy = decltype(make_tiled_copy(
        CopyInst{}, ThreadLayout{},
        Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));

    // TODO(haruhi): The current implementation uses ldmatrix.x4
    // instruction which requires the TileMMA configuration to be
    // fixed as follows. Make it able to be tuned by policy in
    // future implementation.

    using TiledMma =
        TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                 Layout<Shape<Int<kWarpPerRow>, Int<kWarpPerCol>, _1>>,
                 Layout<Shape<_1, _2, _1>>>;

    using SmemLayoutD =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTP>>{}));

    using StoreD_R2S = cell::copy::R2SCopy2D<Element, TiledMma, SmemLayoutD>;
};
}  // namespace tiledcuda::cell::traits
