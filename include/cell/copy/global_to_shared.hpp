#pragma once

#include "cell/copy/constants.hpp"
#include "cell/copy/dyn_copy.hpp"
#include "cell/copy/warp.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

#include <cute/tensor.hpp>

namespace tiledcuda::cell::copy {

using namespace traits;
namespace tl = tile_layout;

template <typename Global_, typename Shared_, const int kThreads_,
          const tl::Layout kType, typename Base>
struct GlobalToSharedLoaderImpl {
    using Global = Global_;
    using Shared = Shared_;
    using DType = typename Global::DType;

    DEVICE void operator()(const DType* src, DType* dst);
};

template <typename Global_, typename Shared_, const int kThreads_,
          typename Base>
struct GlobalToSharedLoaderImpl<Global_, Shared_, kThreads_,
                                tl::Layout::kSwizzledRowMajor, Base> {
    using Global = Global_;
    using Shared = Shared_;
    using DType = typename Global::DType;

    static constexpr int kThreads = kThreads_;

    static constexpr int kRows = Global::kRows;
    static constexpr int kCols = Global::kCols;
    static constexpr int kShmRows = Shared::kRows;
    static constexpr int kShmCols = Shared::kCols;

    // To avoid bank conflict, the shared memory requires a swizzled layout
    static const int kSwizzleMode = kShmCols % 32 ? 1 : 0;
    using Swizzled =
        tl::SwizzledRowMajor<DType, kShmRows, kShmCols, kSwizzleMode>;

    using SrcLayout = tl::RowMajor<kRows, kCols, kCols>;
    using DstLayout = typename Swizzled::SmemLayout;

    // threads in a thread block are laid out as a 2D tile
    // that has a shape of kThreadsRows x kThreadsCols.
    static constexpr int kThreadsCols = kShmCols / Base::kNumPerAccess;
    static constexpr int kThreadsRows = kThreads / kThreadsCols;

    using ThreadLayout = tl::RowMajor<kThreadsRows, kThreadsCols, kThreadsCols>;

    using ValueLayout = Layout<Shape<_1, Int<Base::kNumPerAccess>>>;

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, DType>;
#endif

    using TiledCopy =
        decltype(make_tiled_copy(CopyInst{}, ThreadLayout{}, ValueLayout{}));

    DEVICE void operator()(const DType* src, DType* dst) {
        copy_2d_tile_g2s(src, dst, SrcLayout{}, DstLayout{}, TiledCopy{});
    }
};

template <typename Global_, typename Shared_, const int kThreads_,
          const tl::Layout kType, typename Base>
struct SharedToGlobalStorerImpl {
    using Global = Global_;
    using Shared = Shared_;
    using DType = typename Global::DType;

    DEVICE void operator()(const DType* src, DType* dst);
};

template <typename Global_, typename Shared_, const int kThreads_,
          typename Base>
struct SharedToGlobalStorerImpl<Global_, Shared_, kThreads_,
                                tl::Layout::kSwizzledRowMajor, Base> {
    using Global = Global_;
    using Shared = Shared_;
    using DType = typename Global::DType;

    static constexpr int kThreads = kThreads_;

    static constexpr int kRows = Global::kRows;
    static constexpr int kCols = Global::kCols;
    static constexpr int kShmRows = Shared::kRows;
    static constexpr int kShmCols = Shared::kCols;

    // To avoid bank conflict, the shared memory requires a swizzled layout
    static const int kSwizzleMode = kShmCols % 32 ? 1 : 0;
    using Swizzled =
        tl::SwizzledRowMajor<DType, kShmRows, kShmCols, kSwizzleMode>;

    // using SrcLayout = tl::RowMajor<kRows, kCols, kCols>;
    // using DstLayout = typename Swizzled::SmemLayout;
    using SrcLayout = typename Swizzled::SmemLayout;
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
    using TiledCopy = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, DType>{},
                                               ThreadLayout{}, ValueLayout{}));

    DEVICE void operator()(const DType* src, DType* dst) {
        copy_2d_tile_s2g(src, dst, SrcLayout{}, DstLayout{}, TiledCopy{});
    }
};

template <typename Shared_, typename WarpLayout_,
          const tl::Layout kType_ = tl::Layout::kSwizzledRowMajor>
struct GlobalToSharedLoader {
    using Shared = Shared_;
    using DType = typename Shared::DType;
    using WarpLayout = WarpLayout_;

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr tl::Layout kType = kType_;

    template <typename Global>
    DEVICE void operator()(const Global& src, Shared& dst) {
        const DType* src_ptr = src.data();
        DType* dst_ptr = dst.mutable_data();

        using Loader = GlobalToSharedLoaderImpl<Global, Shared, kThreads, kType,
                                                TraitsBase<DType>>;

        Loader loader;
        loader(src_ptr, dst_ptr);
    }
};

template <typename Shared_, typename WarpLayout_,
          const tl::Layout kType_ = tl::Layout::kSwizzledRowMajor>
struct SharedToGlobalStorer {
    using Shared = Shared_;
    using DType = typename Shared::DType;
    using WarpLayout = WarpLayout_;

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr tl::Layout kType = kType_;

    template <typename Global>
    DEVICE void operator()(const Shared& src, Global& dst) {
        const DType* src_ptr = src.data();
        DType* dst_ptr = dst.mutable_data();

        using Storer = SharedToGlobalStorerImpl<Global, Shared, kThreads, kType,
                                                TraitsBase<DType>>;

        Storer storer;
        storer(src_ptr, dst_ptr);
    }
};

}  // namespace tiledcuda::cell::copy