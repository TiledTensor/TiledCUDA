#pragma once

#include "cell/traits/base.hpp"
#include "types/layout.hpp"

namespace tiledcuda::cell::traits {

using namespace cute;
namespace tl = tiledcuda::tile_layout;

/// @brief Configurations for transfering a single 2D data tile from global
/// memory to shared memory, which include configurating the layout of data tile
/// and thread tile.
/// @tparam Element_: the element type
/// @tparam kThreads: number of threads in a thread block
template <typename Element_, const int kRows_, const int kCols_,
          const int kShmRows_, const int kShmCols_, const int kThreads,
          typename Base = TraitsBase<Element_>>
struct G2S2DCopyTraits : public Base {
    using Element = Element_;

    static constexpr int kRows = kRows_;
    static constexpr int kCols = kCols_;

    static constexpr int kShmRows = kShmRows_;
    static constexpr int kShmCols = kShmCols_;

    using SrcLayout = tl::RowMajor<kRows, kCols, kCols>;

    // To avoid bank conflicts, the shared memory requires a swizzled layout
    static constexpr int kSwizzleMode = kShmCols % 32 ? 1 : 0;
    using Swizzled =
        tl::SwizzledRowMajor<Element, kShmRows, kShmCols, kSwizzleMode>;
    using DstLayout = typename Swizzled::SmemLayout;

    // threads in a thread block are laid out as a 2D tile
    // that has a shape of kThreadsRows x kThreadsCols.
    static constexpr int kThreadsCols = kShmCols / Base::kNumPerAccess;
    static constexpr int kThreadsRows = kThreads / kThreadsCols;
    using ThreadLayout = tl::RowMajor<kThreadsRows, kThreadsCols, kThreadsCols>;

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

/// @brief Configurations for transfering a single 2D data tile from shared
/// memory to global memory, which include configurating the layout of data tile
/// and thread tile.
/// @tparam Element_: the element type
/// @tparam kThreads: number of threads in a thread block
template <typename Element_, const int kRows_, const int kCols_,
          const int kShmRows_, const int kShmCols_, const int kThreads,
          typename Base = TraitsBase<Element_>>
struct S2G2DCopyTraits : public Base {
    using Element = Element_;

    static constexpr int kRows = kRows_;
    static constexpr int kCols = kCols_;

    static constexpr int kShmRows = kShmRows_;
    static constexpr int kShmCols = kShmCols_;

    static constexpr int kSwizzleMode = kShmCols % 32 ? 1 : 0;
    using Swizzled =
        tl::SwizzledRowMajor<Element, kShmRows, kShmCols, kSwizzleMode>;
    using SrcLayout = typename Swizzled::SmemLayout;

    // To avoid bank conflicts, the shared memory requires a swizzled layout
    using DstLayout = tl::RowMajor<kRows, kCols, kCols>;

    // threads in a thread block are laid out as a 2D tile
    // that has a shape of kThreadsRows x kThreadsCols.
    static constexpr int kThreadsCols = kShmCols / Base::kNumPerAccess;
    static constexpr int kThreadsRows = kThreads / kThreadsCols;
    using ThreadLayout = tl::RowMajor<kThreadsRows, kThreadsCols, kThreadsCols>;

    using ValueLayout = Layout<Shape<_1, Int<Base::kNumPerAccess>>>;

    // transfer data from global memory to shared memory has cp.async,
    // while transfer data from shared memory to global memory does not have.
    // for the latter case, the copy instruction should be the default one.
    using TiledCopy = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{}, ThreadLayout{}, ValueLayout{}));
};

}  // namespace tiledcuda::cell::traits
