#pragma once

#include "cell/copy/constants.hpp"
#include "cell/copy/dyn_copy.hpp"
#include "cell/copy/warp.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {
using namespace traits;
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
    // TODO: configuration for how threads are laid out in a single warp,
    // hard-coded it for now. Make it configurable in future implementation.
    // In the row-major layout, columns are contiguous in memory. 4 threads in a
    // warp access data along the continguous dimension (columns), and 8 warps
    // access data along the strided dimension (rows).
    using WarpThreadLayout = tl::RowMajor<8, 4>;
    static constexpr int kThreadRows =
        tl::num_rows<WarpLayout> * tl::num_rows<WarpThreadLayout>;
    static constexpr int kThreadCols =
        tl::num_cols<WarpLayout> * tl::num_cols<WarpThreadLayout>;

    static constexpr int kNumPerAccess = TraitsBase<DType>::kNumPerAccess;
    // Efficient memory access requires specifying the tile shape.
    static_assert(Global::kCols % (kThreadCols * kNumPerAccess) == 0,
                  "The columns of a GlobalTile should be divisible by the "
                  "number of threads per row in the thread layout.");
    static_assert(Global::kRows % kThreadRows == 0,
                  "The rows of a GlobalTile should be divisible by the "
                  "number of threads per column in the thread layout.");

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, DType>;
#endif

    using TiledCopy = decltype(make_tiled_copy(
        CopyInst{}, tl::RowMajor<kThreadRows, kThreadCols>{},
        tl::ColMajor<1, kNumPerAccess>{}));

    DEVICE GlobalToSharedLoaderImpl()
        : src_layout_(typename Global::Layout{}),
          dst_layout_(typename Shared::Layout{}),
          tiled_copy_(TiledCopy{}) {}

    DEVICE void operator()(const DType* src, DType* dst) {
        copy_2d_tile_g2s(src, dst, src_layout_, dst_layout_, tiled_copy_);
    }

  private:
    Global::Layout src_layout_;
    Shared::Layout dst_layout_;
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
    using Global = Global_;
    using Shared = Shared_;
    static_assert(Global::kRows == Shared::kRows &&
                      Global::kCols == Shared::kCols,
                  "Global and shared memory should have the same shape.");

    using DType = Global::DType;

    using WarpLayout = WarpLayout_;
    // TODO: configuration for how threads are laid out in a single warp,
    // hard-coded it for now. Make it configurable in future implementation.
    // In the row-major layout, columns are contiguous in memory. 4 threads in a
    // warp access data along the continguous dimension (columns), and 8 warps
    // access data along the strided dimension (rows).
    using WarpThreadLayout = tl::RowMajor<8, 4>;
    static constexpr int kThreadRows =
        tl::num_rows<WarpLayout> * tl::num_rows<WarpThreadLayout>;
    static constexpr int kThreadCols =
        tl::num_cols<WarpLayout> * tl::num_cols<WarpThreadLayout>;

    static constexpr int kNumPerAccess = TraitsBase<DType>::kNumPerAccess;
    // Efficient memory access requires specifying the tile shape.
    static_assert(Global::kCols % (kThreadCols * kNumPerAccess) == 0,
                  "The columns of a GlobalTile should be divisible by the "
                  "number of threads per row in the thread layout.");
    static_assert(Global::kRows % kThreadRows == 0,
                  "The rows of a GlobalTile should be divisible by the "
                  "number of threads per column in the thread layout.");

    // transfer data from global memory to shared memory has cp.async,
    // while transfer data from shared memory to global memory does not have.
    // for the latter case, the copy instruction should be the default one.
    using TiledCopy =
        decltype(make_tiled_copy(Copy_Atom<DefaultCopy, DType>{},
                                 tl::RowMajor<kThreadRows, kThreadCols>{},
                                 tl::ColMajor<1, kNumPerAccess>{}));

    DEVICE SharedToGlobalStorerImpl()
        : src_layout_(typename Shared::Layout{}),
          dst_layout_(typename Global::Layout{}),
          tiled_copy_(TiledCopy{}) {}

    DEVICE void operator()(const DType* src, DType* dst) {
        copy_2d_tile_s2g(src, dst, src_layout_, dst_layout_, tiled_copy_);
    }

  private:
    Shared::Layout src_layout_;
    Global::Layout dst_layout_;
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
