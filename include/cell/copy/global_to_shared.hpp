#pragma once

#include "cell/copy/mod.hpp"
#include "cell/copy/warp.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

#include <cute/tensor.hpp>

namespace tiledcuda::cell::copy {

using namespace traits;
namespace tl = tile_layout;

template <typename Global_, typename Shared_, const int kRowExec_,
          const int kColExec_, const tl::Layout kType,
          typename Base = TraitsBase<Global_::DType>>
struct GlobalToSharedLoaderImpl {
    using Global = Global_;
    using Shared = Shared_;
    using DType = typename Global::DType;

    static const int kRowExec = kRowExec_;
    static const int kColExec = kColExec_;

    DEVICE void operator()(const DType* src, DType* dst);
};

template <typename Global_, typename Shared_, const int kRowExec_,
          const int kColExec_, const int kThreads_,
          typename Base = TraitsBase<Global_::DType>>
struct GlobalToSharedLoaderImpl<Global_, Shared_, kRowExec_, kColExec_,
                                tl::Layout::kSwizzledRowMajor> {
    using Global = Global_;
    using Shared = Shared_;
    using DType = typename Global::DType;

    static const int kRowExec = kRowExec_;
    static const int kColExec = kColExec_;
    static const int kThreads = kThreads_;

    static const int kShmRows = Shared::kRows;
    static const int kShmCols = Shared::kCols;

    // To avoid bank conflict, the shared memory requires a swizzled layout
    static const int kSwizzleMode = kShmCols % 32 ? 1 : 0;
    using Swizzled =
        tl::SwizzledRowMajor<DType, kShmRows, kShmCols, kSwizzleMode>;

    using SrcLayout = Global::Layout;
    using DstLayout = typename Swizzled::SmemLayout;

    // threads in a thread block are laid out as a 2D tile
    // that has a shape of kThreadsRows x kThreadsCols.
    static constexpr int kThreadsCols = kShmCols / Base::kNumPerAccess;
    static constexpr int kThreadsRows = kThreads / kThreadsCols;

    using ThreadLayout = tl::RowMajor<kThreadsRows, kThreadsCols, kThreadsCols>;

    using ValueLayout = Layout<Shape<_1, Int<Base::kNumPerAccess>>>;

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, Element>;
#endif

    using TiledCopy =
        decltype(make_tiled_copy(CopyInst{}, ThreadLayout{}, ValueLayout{}));

    DEVICE void operator()(const DType* src, DType* dst) {
        int tid = threadIdx.x;

        auto gtile = make_tensor(make_gmem_ptr(src), SrcLayout{});
        auto stile = make_tensor(make_smem_ptr(dst), DstLayout{});

        auto loader = TiledCopy::get_thread_slice(tid);

        auto src_tile = loader.partition_S(gtile);
        auto dst_tile = loader.partition_D(stile);

#pragma unroll
        for (int i = 0; i < int(size<1>(src_tile)); ++i) {
#pragma unroll
            for (int j = 0; j < int(size<2>(src_tile)); ++j) {
                cute::copy(TiledCopy{}, src_tile(_, i, j), dst_tile(_, i, j));
            }
        }
    }
};

template <typename Global_, typename Shared_, typename WarpLayout_>
struct GlobalToSharedLoader {
    using Global = Global_;
    using Shared = Shared_;
    using DType = typename Global::DType;

    using WarpLayout = WarpLayout_;

    DEVICE operator(const Global& src, Shared& dst);
};

}  // namespace tiledcuda::cell::copy