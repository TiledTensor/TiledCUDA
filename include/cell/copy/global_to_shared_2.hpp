#pragma once

#include "types/mod.hpp"

#include <cute/tensor.hpp>

namespace tiledcuda::cell::copy {
using namespace traits;
namespace tl = tile_layout;
using namespace cute;

namespace detail {
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

    static_assert(Global::kRows == Shared::kRows &&
                      Global::kCols == Shared::kCols,
                  "Global and shared memory should have the same shape.");
    static_assert(Global::kType == Shared::kType,
                  "The layout of Global memory and Shared memory tile should "
                  "be the same.");
    static_assert(Global::kType == tl::Layout::kRowMajor,
                  "The layout of Global memory and Shared memory tile should "
                  "be row-major.");
    static_assert(std::is_same_v<typename Shared::DType, DType>,
                  "The data type of Shared and Global must be the same.");

    using WarpThreadLayout = tl::ColMajor<16, 2>;
    static constexpr int kNumPerAccess = TraitsBase<DType>::kNumPerAccess;

    static constexpr int kThreadsRows =
        tl::num_rows<WarpLayout> * tl::num_rows<WarpThreadLayout>;
    static constexpr int kThreadsCols =
        tl::num_cols<WarpLayout> * tl::num_cols<WarpThreadLayout>;

    static constexpr int kRows = Global::kRows;
    static constexpr int kCols = Global::kCols;

    using GlobalLayout =
        cute::Layout<Shape<Int<kRows>, Int<kCols>>, Stride<Int<kCols>, _1>>;

    using LayoutAtom = cute::Layout<Shape<_16, _16>, Stride<_16, _1>>;
    using SharedLayoutNonSwizzled = decltype(tile_to_shape(
        LayoutAtom{}, Shape<Int<kRows>, Int<kCols>>{}, cute::Step<_2, _1>{}));

    // this swizzle function works only for 4-byte data types
    using LayoutAtomSwizzled =
        decltype(composition(Swizzle<2, 3, 3>{}, LayoutAtom{}));
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

    DEVICE void operator()(const DType* src_data, DType* dst_data) {
        TiledCopy tiled_copy;

        int tid = threadIdx.x;

        auto gtile = make_tensor(make_gmem_ptr(src_data), GlobalLayout{});
        auto stile = make_tensor(make_smem_ptr(dst_data), SharedLayout{});

        auto loader = tiled_copy.get_thread_slice(tid);

        auto src = loader.partition_S(gtile);
        auto dst = loader.partition_D(stile);

#pragma unroll
        for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
            for (int j = 0; j < int(size<2>(src)); ++j)
                cute::copy(tiled_copy, src(cute::_, i, j), dst(cute::_, i, j));
    }
};

template <typename Global_, typename Shared_, typename WarpLayout_>
struct GlobalToSharedLoaderImpl2<Global_, Shared_, WarpLayout_,
                                 tl::Layout::kColMajor> {
    using Global = Global_;
    using Shared = Shared_;
    using WarpLayout = WarpLayout_;
    using DType = typename Global::DType;

    static_assert(Global::kRows == Shared::kRows &&
                      Global::kCols == Shared::kCols,
                  "Global and shared memory should have the same shape.");
    static_assert(Global::kType == Shared::kType,
                  "The layout of Global memory and Shared memory tile should "
                  "be the same.");
    static_assert(Global::kType == tl::Layout::kColMajor,
                  "The layout of Global memory and Shared memory tile should "
                  "be column-major.");
    static_assert(std::is_same_v<typename Shared::DType, DType>,
                  "The data type of Shared and Global must be the same.");

    using WarpThreadLayout = tl::RowMajor<2, 16>;
    static constexpr int kNumPerAccess = TraitsBase<DType>::kNumPerAccess;

    static constexpr int kThreadsRows =
        tl::num_rows<WarpLayout> * tl::num_rows<WarpThreadLayout>;
    static constexpr int kThreadsCols =
        tl::num_cols<WarpLayout> * tl::num_cols<WarpThreadLayout>;

    static constexpr int kRows = Global::kRows;
    static constexpr int kCols = Global::kCols;

    using GlobalLayout =
        cute::Layout<Shape<Int<kRows>, Int<kCols>>, Stride<_1, Int<kRows>>>;

    using LayoutAtom = cute::Layout<Shape<_16, _16>, Stride<_1, _16>>;
    // this swizzle function works only for 4-byte data types
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

    DEVICE void operator()(const DType* src_data, DType* dst_data) {
        TiledCopy tiled_copy;
        int tid = threadIdx.x;

        auto gtile = make_tensor(make_gmem_ptr(src_data), GlobalLayout{});
        auto stile = make_tensor(make_smem_ptr(dst_data), SharedLayout{});

        auto loader = tiled_copy.get_thread_slice(tid);

        auto src = loader.partition_S(gtile);
        auto dst = loader.partition_D(stile);

#pragma unroll
        for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
            for (int j = 0; j < int(size<2>(src)); ++j)
                cute::copy(tiled_copy, src(cute::_, i, j), dst(cute::_, i, j));
    }
};

template <typename Shared_, typename Global_, typename WarpLayout_,
          const tl::Layout kType>
struct SharedToGlobalStorerImpl2;

template <typename Shared_, typename Global_, typename WarpLayout_>
struct SharedToGlobalStorerImpl2<Shared_, Global_, WarpLayout_,
                                 tl::Layout::kRowMajor> {
    using Global = Global_;
    using Shared = Shared_;
    using DType = typename Global::DType;
    using WarpLayout = WarpLayout_;

    static_assert(Global::kRows == Shared::kRows &&
                      Global::kCols == Shared::kCols,
                  "Global and shared memory should have the same shape.");
    static_assert(Global::kType == Shared::kType,
                  "The layout of Global memory and Shared memory tile should "
                  "be the same.");
    static_assert(Global::kType == tl::Layout::kRowMajor,
                  "The layout of Global memory and Shared memory tile should "
                  "be row-major.");
    static_assert(std::is_same_v<typename Shared::DType, DType>,
                  "The data type of Shared and Global must be the same.");

    static constexpr int kRows = Global::kRows;
    static constexpr int kCols = Global::kCols;

    static constexpr int kNumPerAccess = TraitsBase<DType>::kNumPerAccess;
    using WarpThreadLayout = tl::ColMajor<16, 2>;

    // thread layout for the entire thread block
    static constexpr int kThreadsRows =
        tl::num_rows<WarpLayout> * tl::num_rows<WarpThreadLayout>;
    static constexpr int kThreadsCols =
        tl::num_cols<WarpLayout> * tl::num_cols<WarpThreadLayout>;

    using BaseTileLayout = cute::Layout<Shape<_16, _16>, Stride<_16, _1>>;
    using SharedLayoutNonSwizzled = decltype(tile_to_shape(
        BaseTileLayout{}, Shape<Int<kRows>, Int<kCols>>{},
        cute::Step<_2, _1>{}));

    // this swizzle function works only for 4-byte data types
    using LayoutAtom =
        decltype(composition(Swizzle<2, 3, 3>{}, BaseTileLayout{}));
    using SharedLayoutSwizzled = decltype(tile_to_shape(
        LayoutAtom{}, Shape<Int<kRows>, Int<kCols>>{}, cute::Step<_2, _1>{}));

    // source layout
    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, SharedLayoutSwizzled,
                           SharedLayoutNonSwizzled>;
    // target layout
    using GlobalLayout =
        cute::Layout<Shape<Int<kRows>, Int<kCols>>, Stride<Int<kCols>, _1>>;

    using ThreadLayout =
        cute::Layout<Shape<Int<kThreadsRows>, Int<kThreadsCols>>,
                     Stride<Int<kThreadsCols>, _1>>;
    using ValueLayout = cute::Layout<Shape<_1, Int<kNumPerAccess>>>;

    // transfer data from global memory to shared memory has cp.async,
    // while transfer data from shared memory to global memory does not have.
    // for the latter case, the copy instruction should be the default one.
    using TiledCopy = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, DType>{},
                                               ThreadLayout{}, ValueLayout{}));

    DEVICE void operator()(const DType* src_data, DType* dst_data) {
        TiledCopy tiled_copy;
        int tid = threadIdx.x;

        auto stile = make_tensor(make_smem_ptr(src_data), SharedLayout{});
        auto gtile = make_tensor(make_gmem_ptr(dst_data), GlobalLayout{});

        auto loader = tiled_copy.get_thread_slice(tid);

        auto src = loader.partition_S(stile);
        auto dst = loader.partition_D(gtile);

#pragma unroll
        for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
            for (int j = 0; j < int(size<2>(src)); ++j) {
                cute::copy(tiled_copy, src(cute::_, i, j), dst(cute::_, i, j));
            }
    }
};
}  // namespace detail

template <typename Shared_, typename WarpLayout_>
struct GlobalToSharedLoader2 {
    using Shared = Shared_;
    using DType = typename Shared::DType;
    using WarpLayout = WarpLayout_;

    static_assert(sizeof(DType) != 4,
                  "Not implemented for data types other than 2 bytes.");

    template <typename Global>
    DEVICE void operator()(const Global& src, Shared& dst) {
        const DType* src_ptr = src.data();
        DType* dst_ptr = dst.mutable_data();

        using Loader =
            detail::GlobalToSharedLoaderImpl2<Global, Shared, WarpLayout_,
                                              Shared::kType>;
        Loader loader;
        loader(src_ptr, dst_ptr);
    }
};

template <typename Shared_, typename WarpLayout_>
struct SharedToGlobalStorer2 {
    using Shared = Shared_;
    using DType = typename Shared::DType;
    using WarpLayout = WarpLayout_;

    static_assert(sizeof(DType) != 4,
                  "Not implemented for data types other than 2 bytes.");

    template <typename Global>
    DEVICE void operator()(const Shared& src_, Global& dst_) {
        const DType* src = src_.data();
        DType* dst = dst_.mutable_data();
        using Storer =
            detail::SharedToGlobalStorerImpl2<Shared, Global, WarpLayout,
                                              Shared::kType>;
        Storer storer;
        storer(src, dst);
    }
};
}  // namespace tiledcuda::cell::copy
