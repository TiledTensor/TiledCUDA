// #pragma once

// #include "cell/copy/mod.hpp"
// #include "cell/traits/base.hpp"

// namespace tiledcuda::cell::traits {
// using namespace cute;
// namespace tl = tile_layout;

// template <typename Element_, typename CtaTileShape,
//           typename WarpArrangement = tiledcuda::cell::TileShape<2, 2>,
//           typename Base = TraitsBase<Element_>>
// struct DynGemmTraits : public Base {
//     using Element = Element_;

//     static constexpr int kTM = dim_size<0, CtaTileShape>;
//     static constexpr int kTN = dim_size<1, CtaTileShape>;
//     static constexpr int kTK = dim_size<2, CtaTileShape>;

//     static_assert(kTM % dim_size<0, WarpArrangement> == 0,
//                   "the M dimension of the CTA tile should be "
//                   "divisible by the "
//                   "number of warps along that that dimension.");
//     static_assert(kTN % dim_size<1, WarpArrangement> == 0,
//                   "the N dimension of the CTA tile should be "
//                   "divisible by the "
//                   "number of warps along that that dimension.");

//     static constexpr int kWarpPerRow = dim_size<0, WarpArrangement>;
//     static constexpr int kWarpPerCol = dim_size<1, WarpArrangement>;
//     // TODO(haruhi): The current implementation uses ldmatrix.x4
//     // instruction which requires the TileMMA configuration to be
//     // fixed as follows. Make it able to be tuned by policy in
//     // future implementation.
//     using TiledMma =
//         TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,  // for ampere
//                  Layout<Shape<Int<kWarpPerRow>, Int<kWarpPerCol>, _1>>,
//                  Tile<Int<16 * kWarpPerRow>, Int<16 * kWarpPerCol>, _16>>;
//     static constexpr int kThreads = size(TiledMma{});
//     static_assert(kThreads == kWarpPerRow * kWarpPerCol * 32);

//     static constexpr int kNumPerAccess = Base::kNumPerAccess;
//     static constexpr int kThreadsPerCol = CeilDiv<kTK, Base::kNumPerAccess>;
//     static constexpr int kThreadsPerRow = CeilDiv<kThreads, kThreadsPerCol>;

//     using SmemLayoutAtom = decltype(composition(
//         Swizzle<2, 3, 3>{}, Layout<Shape<_8, Int<4 * kNumPerAccess>>,
//                                    Stride<Int<4 * kNumPerAccess>, _1>>{}));

//     using SmemLayoutA =
//         decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>,
//         Int<kTK>>{}));
//     using SmemLayoutB =
//         decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>,
//         Int<kTK>>{}));

// #ifdef CP_ASYNC_SM80_ENABLED
//     using CopyInstG2S =
//         Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
// #else
//     using CopyInstG2S = Copy_Atom<DefaultCopy, Element>;
// #endif

//     using TiledCopyG2S = decltype(make_tiled_copy(
//         CopyInstG2S{},
//         Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
//                Stride<Int<kThreadsPerCol>, _1>>{},
//         Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));

//     // copy from shared memory to global memory dose not have cp.async
//     support,
//     // another `TiledCopy` has to be declared and uses `DefaultCopy` as the
//     // `CopyAtom`.
//     using TiledCopyS2G = decltype(make_tiled_copy(
//         Copy_Atom<DefaultCopy, Element>{},
//         Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
//                Stride<Int<kThreadsPerCol>, _1>>{},
//         Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));
//     using SmemLayoutC =
//         decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>,
//         Int<kTN>>{}));
//     using StoreC_R2S = cell::copy::R2SCopy2D<Element, TiledMma, SmemLayoutC>;
// };

// }  // namespace tiledcuda::cell::traits
