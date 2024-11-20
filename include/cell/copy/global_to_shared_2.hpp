#pragma once

#include "cell/copy/mod.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {
using namespace traits;
namespace tl = tile_layout;

template <typename Global_, typename Shared_, typename WarpLayout_,
          const tl::Layout kType>
struct GlobalToSharedLoaderImpl2;

template <typename Global_, typename Shared_, typename WarpLayout_>
struct GlobalToSharedLoaderImpl2<Global_, Shared_, WarpLayout_,
                                 tl::Layout::kRowMajor> {
    using Global = Global_;
    using Shared = Shared_;
    using WarpLayout = WarpLayout_;

    using DType = typename Global::DType;
    using WarpThreadLayout = tl::ColMajor<16, 2>;

    static constexpr int kNumPerAccess = TraitsBase<DType>::kNumPerAccess;
    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;

    static constexpr int kThreadsRows =
        tl::num_rows<WarpLayout> * tl::num_rows<WarpThreadLayout>;
    static constexpr int kThreadsCols =
        tl::num_cols<WarpLayout> * tl::num_cols<WarpThreadLayout>;

    static constexpr int kRows = Global::kRows;
    static constexpr int kCols = Global::kCols;

    using GlobalLayout =
        cute::Layout<Shape<Int<kRows>, Int<kCols>>, Stride<Int<kCols>, _1>>;

    using LayoutAtom = cute::Layout<Shape<_16, _16>, Stride<_16, _1>>;
    using LayoutAtomSwizzled =
        decltype(composition(Swizzle<2, 3, 3>{}, LayoutAtom{}));

    using SharedLayoutNonSwizzled = decltype(tile_to_shape(
        LayoutAtom{}, Shape<Int<kRows>, Int<kCols>>{}, cute::Step<_2, _1>{}));

    using SharedLayoutSwizzled = decltype(tile_to_shape(
        LayoutAtomSwizzled{}, Shape<Int<kRows>, Int<kCols>>{},
        cute::Step<_2, _1>{}));

    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, SharedLayoutSwizzled,
                           SharedLayoutNonSwizzled>;

    using ThreadLayout =
        cute::Layout<Shape<Int<kThreadsRows>, Int<kThreadsCols>>,
                     Stride<Int<kThreadsCols>, _1>>;
    using ValueLayout = cute::Layout<Shape<_1, Int<kNumPerAccess>>>;

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, DType>;
#endif

    using TiledCopy =
        decltype(make_tiled_copy(CopyInst{}, ThreadLayout{}, ValueLayout{}));

    DEVICE void operator()(const DType* src, DType* dst) {
        copy_2d_tile_g2s(src, dst, GlobalLayout{}, SharedLayout{}, TiledCopy{});
    }
};

template <typename Global_, typename Shared_, typename WarpLayout_>
struct GlobalToSharedLoaderImpl2<Global_, Shared_, WarpLayout_,
                                 tl::Layout::kColMajor> {
    using Global = Global_;
    using Shared = Shared_;
    using WarpLayout = WarpLayout_;

    using DType = typename Global::DType;
    using WarpThreadLayout = tl::RowMajor<2, 16>;

    static constexpr int kNumPerAccess = TraitsBase<DType>::kNumPerAccess;
    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;

    static constexpr int kThreadsRows =
        tl::num_rows<WarpLayout> * tl::num_rows<WarpThreadLayout>;
    static constexpr int kThreadsCols =
        tl::num_cols<WarpLayout> * tl::num_cols<WarpThreadLayout>;

    static constexpr int kRows = Global::kRows;
    static constexpr int kCols = Global::kCols;

    using GlobalLayout =
        cute::Layout<Shape<Int<kRows>, Int<kCols>>, Stride<_1, Int<kRows>>>;

    using LayoutAtom = cute::Layout<Shape<_16, _16>, Stride<_1, _16>>;
    using LayoutAtomSwizzled =
        decltype(composition(Swizzle<2, 3, 3>{}, LayoutAtom{}));

    using SharedLayoutNonSwizzled =
        decltype(tile_to_shape(LayoutAtom{}, Shape<Int<kRows>, Int<kCols>>{}));

    using SharedLayoutSwizzled = decltype(tile_to_shape(
        LayoutAtomSwizzled{}, Shape<Int<kRows>, Int<kCols>>{}));

    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, SharedLayoutSwizzled,
                           SharedLayoutNonSwizzled>;

    using ThreadLayout =
        cute::Layout<Shape<Int<kThreadsRows>, Int<kThreadsCols>>,
                     Stride<Int<kThreadsCols>, _1>>;
    using ValueLayout = cute::Layout<Shape<Int<kNumPerAccess>, _1>,
                                     Stride<_1, Int<kNumPerAccess>>>;

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, DType>;
#endif

    using TiledCopy =
        decltype(make_tiled_copy(CopyInst{}, ThreadLayout{}, ValueLayout{}));

    DEVICE void operator()(const DType* src, DType* dst) {
        copy_2d_tile_g2s(src, dst, GlobalLayout{}, SharedLayout{}, TiledCopy{});
    }
};

template <typename Shared_, typename WarpLayout_>
struct GlobalToSharedLoader2 {
    using Shared = Shared_;
    using DType = typename Shared::DType;
    using WarpLayout = WarpLayout_;

    template <typename Global>
    DEVICE void operator()(const Global& src, Shared& dst) {
        const DType* src_ptr = src.data();
        DType* dst_ptr = dst.mutable_data();

        using Loader = GlobalToSharedLoaderImpl2<Global, Shared, WarpLayout_,
                                                 Shared::kType>;

        Loader loader;
        loader(src_ptr, dst_ptr);
    }
};
}  // namespace tiledcuda::cell::copy
