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

    using DType = typename Global::DType;

    using Base = TraitsBase<DType>;
    using WarpLayout = WarpLayout_;

    // interal configuration for the copy kernel
    using WarpThreadLayout = tl::RowMajor<8, 4>;
    // threads in a thread block are laid out as a 2D tile
    // that has a shape of kThreadsRows x kThreadsCols.
    static constexpr int kThreadsCols =
        tl::num_rows<WarpLayout> * tl::num_cols<WarpThreadLayout>;
    static constexpr int kThreadsRows =
        tl::num_cols<WarpLayout> * tl::num_rows<WarpThreadLayout>;

    static_assert(Global::kCols % (kThreadsCols * Base::kNumPerAccess) == 0,
                  "Global memory columns should be divisible by the number of "
                  "threads in a thread block.");
    static_assert(Global::kRows % kThreadsRows == 0,
                  "Global memory rows should be divisible by the number of "
                  "threads in a thread block.");

    using SrcLayout = Global::Layout;
    using DstLayout = Shared::Layout;

    using ThreadLayout = tl::RowMajor<kThreadsRows, kThreadsCols>;
    using ValueLayout = cute::Layout<Shape<_1, Int<Base::kNumPerAccess>>>;

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, DType>;
#endif

    using TiledCopy =
        decltype(make_tiled_copy(CopyInst{}, ThreadLayout{}, ValueLayout{}));

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

    using DType = typename Global::DType;
    using Base = TraitsBase<DType>;
    using WarpLayout = WarpLayout_;

    // interal configuration for the copy kernel
    using WarpThreadLayout = tl::RowMajor<8, 4>;
    // threads in a thread block are laid out as a 2D tile
    // that has a shape of kThreadsRows x kThreadsCols.
    static constexpr int kThreadsCols =
        tl::num_rows<WarpLayout> * tl::num_cols<WarpThreadLayout>;
    static constexpr int kThreadsRows =
        tl::num_cols<WarpLayout> * tl::num_rows<WarpThreadLayout>;
    static_assert(Global::kCols % (kThreadsCols * Base::kNumPerAccess) == 0,
                  "Global memory columns should be divisible by the number of "
                  "threads in a thread block.");
    static_assert(Global::kRows % kThreadsRows == 0,
                  "Global memory rows should be divisible by the number of "
                  "threads in a thread block.");

    using ThreadLayout = tl::RowMajor<kThreadsRows, kThreadsCols>;
    using ValueLayout = Layout<Shape<_1, Int<Base::kNumPerAccess>>>;
    // transfer data from global memory to shared memory has cp.async,
    // while transfer data from shared memory to global memory does not have.
    // for the latter case, the copy instruction should be the default one.
    using TiledCopy = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, DType>{},
                                               ThreadLayout{}, ValueLayout{}));

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
    using DType = typename Shared::DType;
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
    using DType = typename Shared::DType;
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
