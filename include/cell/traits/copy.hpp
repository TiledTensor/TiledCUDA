#pragma once

#include "cell/traits/base.hpp"
#include "layout.hpp"

namespace tiledcuda::cell::traits {

/// @brief Configurations for layout and shape of the 2D data tile.
/// @tparam Element_: the element type
/// @tparam kThreads
template <typename Element_, const int kRows_, const int kCols_,
          const int kShmRows_, const int kShmCols_, const int kThreads,
          typename Base = TraitsBase<Element_>>
struct G2S2DCopyTraits : public Base {
    using Element = Element_;

    static constexpr int kRows = kRows_;
    static constexpr int kCols = kCols_;

    static constexpr int kShmRows = kShmRows_;
    static constexpr int kShmCols = kShmCols_;

    // The layout for the source 2D data tile
    // source place is the global memory
    // using SrcLayout = decltype(make_row_major_layout(kRows, kCols, kCols));

    // The layout for the shared memory 2D data tile
    // the destination place is the shared memory
    // using TrgLayout =
    //     decltype(make_row_major_layout(kShmRows, kShmCols, kShmCols));

    using SrcLayout = RowMajor<kRows, kCols, kCols>;

    // To avoid bank conflicts, the shared memory requires a swizzled layout
    using DstLayout = RowMajor<kShmRows, kShmCols, kShmCols>;

    // static constexpr int kSwizzle = (kShmCols == 32 ? 2 : 3);
    // using SmemLayoutAtom = decltype(composition(
    //     Swizzle<kSwizzle, 3, 3>{},
    //     Layout<Shape<_8, Int<kTK>>, Stride<Int<kTK>, _1>>{}));
    // using SmemLayoutA =
    //     decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>,
    //     Int<kTK>>{}));

    // threads in a thread block are laid out as a 2D tile
    // that has a shape of kThreadsRows x kThreadsCols.
    static constexpr int kThreadsCols = kShmCols / Base::kNumPerAccess;
    static constexpr int kThreadsRows = kThreads / kThreadsCols;
    using ThreadLayout = RowMajor<kThreadsRows, kThreadsCols, kThreadsCols>;

    using ValueLayout = Layout<Shape<_1, Int<Base::kNumPerAccess>>>;

#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, Element>;
#endif

    using TiledCopy =
        decltype(make_tiled_copy(CopyInst{}, ThreadLayout{}, ValueLayout{}));
};

/// @brief Configurations for layout and shape of the 2D data tile.
/// @tparam Element_: the element type
/// @tparam kThreads
template <typename Element_, const int kRows_, const int kCols_,
          const int kShmRows_, const int kShmCols_, const int kThreads,
          typename Base = TraitsBase<Element_>>
struct S2G2DCopyTraits : public Base {
    using Element = Element_;

    static constexpr int kRows = kRows_;
    static constexpr int kCols = kCols_;

    static constexpr int kShmRows = kShmRows_;
    static constexpr int kShmCols = kShmCols_;

    // The layout for the source 2D data tile
    // source place is the global memory
    // using SrcLayout = decltype(make_row_major_layout(kRows, kCols, kCols));

    // The layout for the shared memory 2D data tile
    // the destination place is the shared memory
    // using TrgLayout =
    //     decltype(make_row_major_layout(kShmRows, kShmCols, kShmCols));

    using SrcLayout = RowMajor<kShmRows, kShmCols, kShmCols>;

    // To avoid bank conflicts, the shared memory requires a swizzled layout
    using DstLayout = RowMajor<kRows, kCols, kCols>;

    // threads in a thread block are laid out as a 2D tile
    // that has a shape of kThreadsRows x kThreadsCols.
    static constexpr int kThreadsCols = kShmCols / Base::kNumPerAccess;
    static constexpr int kThreadsRows = kThreads / kThreadsCols;
    using ThreadLayout = RowMajor<kThreadsRows, kThreadsCols, kThreadsCols>;

    using ValueLayout = Layout<Shape<_1, Int<Base::kNumPerAccess>>>;

    // transfer data from global memory to shared memory has cp.async,
    // while transfer data from shared memory to global memory does not have.
    // for the latter case, the copy instruction should be the default one.
    using TiledCopy = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{}, ThreadLayout{}, ValueLayout{}));
};

}  // namespace tiledcuda::cell::traits
