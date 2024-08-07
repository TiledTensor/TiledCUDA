#pragma once

#include "cell/copy/mod.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {
using namespace atom;
namespace tl = tile_layout;

template <typename Global, typename Shared, const int kRowExec,
          const int kColExec, const tl::Layout kType>
struct GlobalToSharedLoaderImpl {
    using DType = Global::DType;

    DEVICE void operator()(const DType* src, DType* dst);
};

template <typename Global_, typename Shared_, const int kRowExec_,
          const int kColExec_>
struct GlobalToSharedLoaderImpl<Global_, Shared_, kRowExec_, kColExec_,
                                tl::Layout::kRowMajor>
    : public GlobalToSharedBaseTileLoader<Global_, Shared_,
                                          tl::Layout::kRowMajor> {
    using Global = Global_;
    using Shared = Shared_;

    static_assert(Global::kRows == Shared::kRows &&
                      Global::kCols == Shared::kCols,
                  "Global and shared memory should have the same shape.");
    static_assert(Global::kType == Shared::kType,
                  "The layout of Global memory and Shared memory tile should "
                  "be the same.");
    static_assert(Global::kType == tl::Layout::kRowMajor,
                  "The layout of Global memory and Shared memory tile should "
                  "be row-major.");

    using DType = Global::DType;

    using LoadBase =
        GlobalToSharedBaseTileLoader<Global, Shared, tl::Layout::kRowMajor>;

    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    // strides to iterate over each 16x16 `BaseTile` in the shared memory
    static constexpr int kSrcRstride = BaseShape::kRows * Global::kCols;
    static constexpr int kDstRstride = BaseShape::kRows * Shared::kCols;

    static constexpr int kCstride = BaseShape::kCols;

    static constexpr int kNumPerAccess = LoadBase::kNumPerAccess;

    DEVICE void operator()(const DType* src, DType* dst) {
        int lane_row = this->lane_row_id();
        int lane_col = this->lane_col_id() * kNumPerAccess;

        const int tid = 18;
        if (thread(tid)) {
            printf("use swizzled layout = %d\n", Shared::kSwizzled);

            printf("exec count = [%d, %d]\n", kRowExec, kColExec);
            printf("lane_row = %d, lane_col = %d\n", lane_row, lane_col);

            printf("kSrcRstride = %d\n", kSrcRstride);
            printf("kDstRstride = %d\n", kDstRstride);
        }

        int src_lane_offset = src_layout_(lane_row, lane_col);
        int dst_lane_offset = dst_layout_(lane_row, lane_col);

        int src_offset = 0, dst_offset = 0;
        for (int i = 0; i < kRowExec; ++i) {
            for (int j = 0; j < kColExec; ++j) {
                src_offset = i * kSrcRstride + j * kCstride + src_lane_offset;
                dst_offset = i * kDstRstride + j * kCstride + src_lane_offset;

                if (thread(tid)) {
                    printf("\n =======  loader final [%d, %d] ======\n", i, j);
                    printf("src_offset = %d, dst_offset = %d\n", src_offset,
                           dst_offset);
                }

                this->copy(src + src_offset, dst + dst_offset);

                if (thread(tid)) {
                    auto data = reinterpret_cast<const __half*>(dst);

                    for (int i = 0; i < Shared::kRows; ++i) {
                        for (int j = 0; j < Shared::kCols; ++j) {
                            printf("%.0f, ",
                                   __half2float(data[i * Shared::kCols + j]));
                        }
                        printf("\n");
                    }
                }
            }
        }
    }

  private:
    typename LoadBase::BaseTileGlobalLayout src_layout_;
    typename LoadBase::BaseTileSharedLayout dst_layout_;
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
    //     using Shared = Shared_;
    //     using Global = Global_;
    //     static_assert(Global::kRows == Shared::kRows &&
    //                       Global::kCols == Shared::kCols,
    //                   "Global and shared memory should have the same
    //                   shape.");

    //     using DType = Global::DType;
    //     static constexpr int kSizeOfTypeBits = int(sizeof(DType) * 8);

    //     using WarpLayout = WarpLayout_;

    //     using WarpThreadLayout = tl::RowMajor<16, 2>;
    //     static constexpr int kThreadRows =
    //         tl::num_rows<WarpLayout> * tl::num_rows<WarpThreadLayout>;
    //     static constexpr int kThreadCols =
    //         tl::num_cols<WarpLayout> * tl::num_cols<WarpThreadLayout>;

    //     using BaseShape = traits::BaseTileShape<DType>;
    //     static constexpr int kWarpTileRows =
    //         tl::num_rows<WarpLayout> * BaseShape::kRows;
    //     static constexpr int kNumPerAccess =
    //         traits::TraitsBase<DType>::kNumPerAccess;
    //     static constexpr int kWarpTileCols = kThreadCols * kNumPerAccess;

    //     static_assert(Shared::kRows % kWarpTileRows == 0,
    //                   "The number of shared memory rows must be divisible by
    //                   the " "warp tile row.");
    //     static_assert(Shared::kCols % kWarpTileCols == 0,
    //                   "The number of shared memory columns must be divisible
    //                   by " "the warp tile column.");

    //     static constexpr int kRowExec = Shared::kRows / kWarpTileRows;
    //     static constexpr int kColExec = Shared::kCols / kWarpTileCols;

    //     // Efficient memory access requires specifying the tile shape.
    //     static_assert(Global::kCols % kWarpTileCols == 0,
    //                   "The columns of a GlobalTile should be divisible by the
    //                   " "number of threads per row in the thread layout.");
    //     static_assert(Global::kRows % kThreadRows == 0,
    //                   "The rows of a GlobalTile should be divisible by the "
    //                   "number of threads per column in the thread layout.");

    //     using SharedLayout = tl::SharedLayoutWrapper<Shared>::Layout;

    //     using GlobalLayout =
    //         cute::Layout<Shape<Int<kWarpTileRows>, Int<kWarpTileCols>>,
    //                      Stride<Int<Global::kRowStride>, _1>>;

    //     // The macro kernel decompose the entire copy operation into 16x16
    //     // BaseTile. The strides are computed to advance the pointer to the
    //     next
    //     // BaseTile for global and shared memory.
    //     static constexpr int kSrcRstride = kWarpTileRows *
    //     Shared::kRowStride; static constexpr int kDstRstride = kWarpTileRows
    //     * Global::kRowStride;

    //     // transfer data from global memory to shared memory has cp.async,
    //     // while transfer data from shared memory to global memory does not
    //     // have. For the latter case, the copy instruction should be the
    //     // default one.
    //     using TiledCopy = decltype(make_tiled_copy(
    //         Copy_Atom<DefaultCopy, DType>{},
    //         cute::Layout<Shape<Int<kThreadRows>, Int<kThreadCols>>,
    //                      Stride<Int<kThreadCols>, _1>>{},
    //         cute::Layout<Shape<_1, Int<kNumPerAccess>>>{}));

    //     DEVICE SharedToGlobalStorerImpl()
    //         : src_layout_(SharedLayout{}),
    //           dst_layout_(GlobalLayout{}),
    //           tiled_copy_(TiledCopy{}) {}

    //     DEVICE void operator()(const DType* src, DType* dst) {
    //         // if (thread0()) {
    //         //     auto src_ = reinterpret_cast<const __half*>(src);
    //         //     printf("\n =========  storer initial =========== \n");

    //         //     for (int i = 0; i < Shared::kRows; ++i) {
    //         //         for (int j = 0; j < Shared::kCols; ++j) {
    //         //             printf("%.0f, ", __half2float(src_[i *
    //         Shared::kCols +
    //         //             j]));
    //         //         }
    //         //         printf("\n");
    //         //     }
    //         // }

    //         int off_src = 0, off_dst = 0;
    //         for (int i = 0; i < kRowExec; ++i) {
    //             for (int j = 0; j < kColExec; ++j) {
    //                 off_src = i * kSrcRstride + j * kWarpTileCols;
    //                 off_dst = i * kDstRstride + j * kWarpTileCols;

    //                 copy_2d_tile_s2g(src + off_src, dst + off_dst,
    //                 src_layout_,
    //                                  dst_layout_, tiled_copy_);
    //             }
    //         }
    //     }

    //   private:
    //     SharedLayout src_layout_;
    //     GlobalLayout dst_layout_;
    //     TiledCopy tiled_copy_;
};

template <typename Shared_, typename WarpLayout_,
          typename Base = warp::CopyBase<WarpLayout_, WarpReuse::kCont>>
struct GlobalToSharedLoader : public Base {
    using Shared = Shared_;
    using DType = Shared::DType;
    using WarpLayout = WarpLayout_;

    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRowExec =
        Base::template row_exec_count<BaseShape, Shared::kRows>();
    static constexpr int kColExec =
        Base::template col_exec_count<BaseShape, Shared::kCols>();

    template <typename Global>
    DEVICE void operator()(const Global& src, Shared& dst) {
        static_assert(Shared::kNumel == Global::kNumel,
                      "Global and shared memory should have the same shape.");
        static_assert(
            Global::kRows == Shared::kRows && Global::kCols == Shared::kCols,
            "Global and shared memory should have the same shape.");

        const DType* src_ptr = src.data();
        DType* dst_ptr = dst.mutable_data();

        int offset_src = Base::template get_warp_offset<Global>();
        int offset_dst = Base::template get_warp_offset<Shared>();

        using Loader = GlobalToSharedLoaderImpl<Global, Shared, kRowExec,
                                                kColExec, Shared::kType>;

        Loader loader;
        loader(src_ptr + offset_src, dst_ptr + offset_dst);
    }
};

template <typename Shared_, typename WarpLayout_>
struct SharedToGlobalStorer {
    using Shared = Shared_;
    using DType = Shared::DType;
    using WarpLayout = WarpLayout_;

    static constexpr tl::Layout kType = Shared::kType;

    template <typename Global>
    DEVICE void operator()(const Shared& src, Global& dst) {
        const DType* src_ptr = src.data();
        DType* dst_ptr = dst.mutable_data();

        using Storer =
            SharedToGlobalStorerImpl<Global, Shared, WarpLayout, Shared::kType>;

        // Storer storer;
        // storer(src_ptr, dst_ptr);
    }
};

}  // namespace tiledcuda::cell::copy
