#pragma once

#include "cell/copy/constants.hpp"
#include "cell/copy/dyn_copy.hpp"
#include "cell/copy/warp.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {
namespace tl = tile_layout;

template <typename Global, typename Shared, typename WarpLayout,
          const tl::Layout kType>
struct GlobalToSharedLoaderImpl {
    using DType = Global::DType;

    DEVICE void operator()(const DType* src, DType* dst);
};

template <typename Global_, typename Shared_, typename WarpLayout_>
struct GlobalToSharedLoaderImpl<Global_, Shared_, WarpLayout_,
                                tl::Layout::kRowMajor> {
    using Global = Global_;
    using Shared = Shared_;
    static_assert(Global::kRows == Shared::kRows &&
                      Global::kCols == Shared::kCols,
                  "Global and shared memory should have the same shape.");

    using DType = Global::DType;
    using WarpLayout = WarpLayout_;

    // The macro kernel breaks down the entire copy operation into iterations
    // over 16x16 BaseTiles. To transfer a single BaseTile, threads in a warp
    // are arranged in a 16x2 row-major layout. Each thread uses 128-bit data in
    // a single access.
    using WarpThreadLayout = tl::RowMajor<16, 2>;
    static constexpr int kThreadRows =
        tl::num_rows<WarpLayout> * tl::num_rows<WarpThreadLayout>;
    static constexpr int kThreadCols =
        tl::num_cols<WarpLayout> * tl::num_cols<WarpThreadLayout>;

    using BaseShape = traits::BaseTileShape<DType>;
    static constexpr int kWarpTileRows =
        tl::num_rows<WarpLayout> * BaseShape::kRows;
    static constexpr int kNumPerAccess =
        traits::TraitsBase<DType>::kNumPerAccess;
    static constexpr int kWarpTileCols = kThreadCols * kNumPerAccess;

    static_assert(Shared::kRows % kWarpTileRows == 0,
                  "The number of shared memory rows must be divisible by the "
                  "warp tile row.");
    static_assert(Shared::kCols % kWarpTileCols == 0,
                  "The number of shared memory columns must be divisible by "
                  "the warp tile column.");

    static constexpr int kRowExec = Global::kRows / kWarpTileRows;
    static constexpr int kColExec = Global::kCols / kWarpTileCols;

    // Efficient memory access requires specifying the tile shape.
    static_assert(Global::kCols % (kThreadCols * kNumPerAccess) == 0,
                  "The columns of a GlobalTile should be divisible by the "
                  "number of threads per row in the thread layout.");
    static_assert(Global::kRows % kThreadRows == 0,
                  "The rows of a GlobalTile should be divisible by the number "
                  "of threads per column in the thread layout.");

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, DType>;
#endif

    // FIXME(haruhi): Swizzled shared memory layout is not yet supported.
    // The current implementation relies on CuTe's API and uses CuTe's layout.
    // Refactor gradually to remove this dependency in the future.
    using GlobalLayout =
        cute::Layout<Shape<Int<kWarpTileRows>, Int<kWarpTileCols>>,
                     Stride<Int<Global::kRowStride>, _1>>;
    using SharedLayout =
        cute::Layout<Shape<Int<kWarpTileRows>, Int<kWarpTileCols>>,
                     Stride<Int<Shared::kRowStride>, _1>>;
    using TiledCopy = decltype(make_tiled_copy(
        CopyInst{},
        cute::Layout<Shape<Int<kThreadRows>, Int<kThreadCols>>,
                     Stride<Int<kThreadCols>, _1>>{},
        cute::Layout<Shape<_1, Int<kNumPerAccess>>>{}));

    // The macro kernel breaks down the entire copy operation into 16x16
    // BaseTiles.
    // Strides are computed to advance the pointers to the next BaseTile in
    // global and shared memory.
    static constexpr int kSrcRstride = kWarpTileRows * Global::kRowStride;
    static constexpr int kDstRstride = kWarpTileRows * Shared::kRowStride;

    DEVICE GlobalToSharedLoaderImpl()
        : src_layout_(GlobalLayout{}),
          dst_layout_(SharedLayout{}),
          tiled_copy_(TiledCopy{}) {}

    DEVICE void operator()(const DType* src, DType* dst) {
        int off_src = 0, off_dst = 0;
        for (int i = 0; i < kRowExec; ++i) {
            for (int j = 0; j < kColExec; ++j) {
                off_src = i * kSrcRstride + j * kWarpTileCols;
                off_dst = i * kDstRstride + j * kWarpTileCols;

                copy_2d_tile_g2s(src + off_src, dst + off_dst, src_layout_,
                                 dst_layout_, tiled_copy_);
            }
        }
    }

  private:
    GlobalLayout src_layout_;
    SharedLayout dst_layout_;
    TiledCopy tiled_copy_;
};

template <typename Global, typename Shared, typename WarpLayout,
          const tl::Layout kType>
struct SharedToGlobalStorerImpl {
    using DType = Global::DType;

    DEVICE void operator()(const DType* src, DType* dst);
};

template <typename Global_, typename Shared_, typename WarpLayout_>
struct SharedToGlobalStorerImpl<Global_, Shared_, WarpLayout_,
                                tl::Layout::kRowMajor> {
    using Shared = Shared_;
    using Global = Global_;
    static_assert(Global::kRows == Shared::kRows &&
                      Global::kCols == Shared::kCols,
                  "Global and shared memory should have the same shape.");

    using DType = Global::DType;
    using WarpLayout = WarpLayout_;

    using WarpThreadLayout = tl::RowMajor<16, 2>;
    static constexpr int kThreadRows =
        tl::num_rows<WarpLayout> * tl::num_rows<WarpThreadLayout>;
    static constexpr int kThreadCols =
        tl::num_cols<WarpLayout> * tl::num_cols<WarpThreadLayout>;

    using BaseShape = traits::BaseTileShape<DType>;
    static constexpr int kWarpTileRows =
        tl::num_rows<WarpLayout> * BaseShape::kRows;
    static constexpr int kNumPerAccess =
        traits::TraitsBase<DType>::kNumPerAccess;
    static constexpr int kWarpTileCols = kThreadCols * kNumPerAccess;

    static_assert(Shared::kRows % kWarpTileRows == 0,
                  "The number of shared memory rows must be divisible by the "
                  "warp tile row.");
    static_assert(Shared::kCols % kWarpTileCols == 0,
                  "The number of shared memory columns must be divisible by "
                  "the warp tile column.");

    static constexpr int kRowExec = Shared::kRows / kWarpTileRows;
    static constexpr int kColExec = Shared::kCols / kWarpTileCols;

    // Efficient memory access requires specifying the tile shape.
    static_assert(Global::kCols % kWarpTileCols == 0,
                  "The columns of a GlobalTile should be divisible by the "
                  "number of threads per row in the thread layout.");
    static_assert(Global::kRows % kThreadRows == 0,
                  "The rows of a GlobalTile should be divisible by the "
                  "number of threads per column in the thread layout.");

    // FIXME(haruhi): since the current implementation relies on CuTe's API,
    // this is CuTe's layout. Refactor gradually to remove this dependency in
    // the future.
    using GlobalLayout =
        cute::Layout<Shape<Int<kWarpTileRows>, Int<kWarpTileCols>>,
                     Stride<Int<Global::kRowStride>, _1>>;
    // FIXME(haruhi): Swizzled shared memory layout is not supported yet.
    // The current implementation relies on CuTe's API, this is CuTe's layout.
    // Refactor gradually to remove this dependency in the future.
    using SharedLayout =
        cute::Layout<Shape<Int<kWarpTileRows>, Int<kWarpTileCols>>,
                     Stride<Int<Shared::kRowStride>, _1>>;

    // The macro kernel decompose the entire copy operation into 16x16 BaseTile.
    // The strides are computed to advance the pointer to the next BaseTile for
    // global and shared memory.
    static constexpr int kSrcRstride = kWarpTileRows * Global::kRowStride;
    static constexpr int kDstRstride = kWarpTileRows * Shared::kRowStride;

    // transfer data from global memory to shared memory has cp.async,
    // while transfer data from shared memory to global memory does not
    // have. For the latter case, the copy instruction should be the
    // default one.
    using TiledCopy = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, DType>{},
        cute::Layout<Shape<Int<kThreadRows>, Int<kThreadCols>>,
                     Stride<Int<kThreadCols>, _1>>{},
        cute::Layout<Shape<_1, Int<kNumPerAccess>>>{}));

    DEVICE SharedToGlobalStorerImpl()
        : src_layout_(SharedLayout{}),
          dst_layout_(GlobalLayout{}),
          tiled_copy_(TiledCopy{}) {}

    DEVICE void operator()(const DType* src, DType* dst) {
        int off_src = 0, off_dst = 0;
        for (int i = 0; i < kRowExec; ++i) {
            for (int j = 0; j < kColExec; ++j) {
                off_src = i * kSrcRstride + j * kWarpTileCols;
                off_dst = i * kDstRstride + j * kWarpTileCols;

                copy_2d_tile_s2g(src + off_src, dst + off_dst, src_layout_,
                                 dst_layout_, tiled_copy_);
            }
        }
    }

  private:
    SharedLayout src_layout_;
    GlobalLayout dst_layout_;
    TiledCopy tiled_copy_;
};

template <typename Shared_, typename WarpLayout_,
          const tl::Layout kType_ = tl::Layout::kRowMajor>
struct GlobalToSharedLoader {
    using Shared = Shared_;
    using DType = Shared::DType;
    using WarpLayout = WarpLayout_;

    static constexpr tl::Layout kType = kType_;

    template <typename Global>
    DEVICE void operator()(const Global& src, Shared& dst) {
        static_assert(Shared::kNumel == Global::kNumel,
                      "Global and shared memory should have the same shape.");
        static_assert(
            Global::kRows == Shared::kRows && Global::kCols == Shared::kCols,
            "Global and shared memory should have the same shape.");

        const DType* src_ptr = src.data();
        DType* dst_ptr = dst.mutable_data();

        using Loader =
            GlobalToSharedLoaderImpl<Global, Shared, WarpLayout, kType>;

        Loader loader;
        loader(src_ptr, dst_ptr);
    }
};

template <typename Shared_, typename WarpLayout_,
          const tl::Layout kType_ = tl::Layout::kRowMajor>
struct SharedToGlobalStorer {
    using Shared = Shared_;
    using DType = Shared::DType;
    using WarpLayout = WarpLayout_;

    static constexpr tl::Layout kType = kType_;

    template <typename Global>
    DEVICE void operator()(const Shared& src, Global& dst) {
        const DType* src_ptr = src.data();
        DType* dst_ptr = dst.mutable_data();

        using Storer =
            SharedToGlobalStorerImpl<Global, Shared, WarpLayout, kType>;

        Storer storer;
        storer(src_ptr, dst_ptr);
    }
};

}  // namespace tiledcuda::cell::copy
