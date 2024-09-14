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

    static_assert(std::is_same_v<typename Shared::DType, DType>,
                  "The data type of Shared and Global must be the same.");

    using LoadBase =
        GlobalToSharedBaseTileLoader<Global, Shared, tl::Layout::kRowMajor>;
    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    static constexpr int kNumPerAccess = LoadBase::kNumPerAccess;

    // strides to iterate over each 16x16 `BaseTile` in the shared memory
    static constexpr int kSrcRstride = BaseShape::kRows * Global::kRowStride;
    static constexpr int kDstRstride = BaseShape::kRows * Shared::kRowStride;
    static constexpr int kCstride = BaseShape::kCols;

    DEVICE void operator()(const DType* src, DType* dst) {
        int lane_row = this->lane_row_id();
        int lane_col = this->lane_col_id() * kNumPerAccess;

        int src_lane_offset = src_layout_(lane_row, lane_col);
        int dst_lane_offset = dst_layout_(lane_row, lane_col);

        int src_offset = 0, dst_offset = 0;
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                src_offset = i * kSrcRstride + j * kCstride + src_lane_offset;
                dst_offset = i * kDstRstride + j * kCstride + dst_lane_offset;

                this->copy(src + src_offset, dst + dst_offset);
            }
        }
    }

  private:
    typename LoadBase::BaseTileGlobalLayout src_layout_;
    typename LoadBase::BaseTileSharedLayout dst_layout_;
};

template <typename Global_, typename Shared_, const int kRowExec_,
          const int kColExec_>
struct GlobalToSharedLoaderImpl<Global_, Shared_, kRowExec_, kColExec_,
                                tl::Layout::kColMajor>
    : public GlobalToSharedBaseTileLoader<Global_, Shared_,
                                          tl::Layout::kColMajor> {
    using Global = Global_;
    using Shared = Shared_;

    static_assert(Global::kRows == Shared::kRows &&
                      Global::kCols == Shared::kCols,
                  "Global and shared memory should have the same shape.");
    static_assert(Global::kType == Shared::kType,
                  "The layout of Global memory and Shared memory tile should "
                  "be the same.");
    static_assert(Global::kType == tl::Layout::kColMajor,
                  "The layout of Global memory and Shared memory tile should "
                  "be column-major.");

    using DType = Global::DType;

    static_assert(std::is_same_v<typename Shared::DType, DType>,
                  "The data type of Shared and Global must be the same.");

    using LoadBase =
        GlobalToSharedBaseTileLoader<Global, Shared, tl::Layout::kColMajor>;
    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    static constexpr int kNumPerAccess = LoadBase::kNumPerAccess;

    // strides to iterate over each 16x16 `BaseTile` in the shared memory
    static constexpr int kRstride = BaseShape::kRows;
    static constexpr int kSrcCstride = BaseShape::kCols * Global::kColStride;
    static constexpr int kDstCstride = BaseShape::kCols * Shared::kColStride;

    DEVICE void operator()(const DType* src, DType* dst) {
        int lane_row = this->lane_row_id() * kNumPerAccess;
        int lane_col = this->lane_col_id();

        int src_lane_offset = src_layout_(lane_row, lane_col);
        int dst_lane_offset = dst_layout_(lane_row, lane_col);

        int src_offset = 0, dst_offset = 0;

        // In the column-major layout, rows are contiguous in memory, we
        // made the inner loop iterate over rows
#pragma unroll
        for (int i = 0; i < kColExec; ++i) {
#pragma unroll
            for (int j = 0; j < kRowExec; ++j) {
                src_offset = j * kRstride + i * kSrcCstride + src_lane_offset;
                dst_offset = j * kRstride + i * kDstCstride + dst_lane_offset;

                this->copy(&src[src_offset], &dst[dst_offset]);
            }
        }
    }

  private:
    typename LoadBase::BaseTileGlobalLayout src_layout_;
    typename LoadBase::BaseTileSharedLayout dst_layout_;
};

template <typename Shared, typename Global, const int kRowExec_,
          const int kColExec_, const tl::Layout kType>
struct SharedToGlobalStorerImpl {
    using DType = Global::DType;

    DEVICE void operator()(const DType* src, DType* dst);
};

template <typename Shared_, typename Global_, const int kRowExec_,
          const int kColExec_>
struct SharedToGlobalStorerImpl<Shared_, Global_, kRowExec_, kColExec_,
                                tl::Layout::kRowMajor>
    : public SharedToGlobalBaseTileStorer<Shared_, Global_,
                                          tl::Layout::kRowMajor> {
    using Shared = Shared_;
    using Global = Global_;
    static_assert(Global::kRows == Shared::kRows &&
                      Global::kCols == Shared::kCols,
                  "Global and shared memory should have the same shape.");
    static_assert(Global::kType == Shared::kType,
                  "The layout of Global memory and Shared memory tile should "
                  "be the same.");
    static_assert(Global::kType == tl::Layout::kRowMajor,
                  "The layout of Global memory and Shared memory tile should "
                  "be row-major.");

    using DType = Shared::DType;

    static_assert(std::is_same_v<typename Global::DType, DType>,
                  "The data type of Shared and Global must be the same.");

    using StoreBase =
        SharedToGlobalBaseTileStorer<Shared, Global, tl::Layout::kRowMajor>;

    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    // strides to iterate over each 16x16 `BaseTile` in the shared memory
    static constexpr int kSrcRstride = BaseShape::kRows * Shared::kRowStride;
    static constexpr int kDstRstride = BaseShape::kRows * Global::kRowStride;

    static constexpr int kCstride = BaseShape::kCols;

    static constexpr int kNumPerAccess = StoreBase::kNumPerAccess;

    DEVICE void operator()(const DType* src, DType* dst) {
        int lane_row = this->lane_row_id();
        int lane_col = this->lane_col_id() * kNumPerAccess;

        int src_lane_offset = src_layout_(lane_row, lane_col);
        int dst_lane_offset = dst_layout_(lane_row, lane_col);

        int src_offset = 0, dst_offset = 0;
        for (int i = 0; i < kRowExec; ++i) {
            for (int j = 0; j < kColExec; ++j) {
                src_offset = i * kSrcRstride + j * kCstride + src_lane_offset;
                dst_offset = i * kDstRstride + j * kCstride + dst_lane_offset;

                this->copy(src + src_offset, dst + dst_offset);
            }
        }
    }

  private:
    typename StoreBase::BaseTileGlobalLayout src_layout_;
    typename StoreBase::BaseTileSharedLayout dst_layout_;
};

template <typename Shared_, typename Global_, const int kRowExec_,
          const int kColExec_>
struct SharedToGlobalStorerImpl<Shared_, Global_, kRowExec_, kColExec_,
                                tl::Layout::kColMajor>
    : public SharedToGlobalBaseTileStorer<Shared_, Global_,
                                          tl::Layout::kColMajor> {
    using Shared = Shared_;
    using Global = Global_;
    static_assert(Global::kRows == Shared::kRows &&
                      Global::kCols == Shared::kCols,
                  "Global and shared memory should have the same shape.");
    static_assert(Global::kType == Shared::kType,
                  "The layout of Global memory and Shared memory tile should "
                  "be the same.");
    static_assert(Global::kType == tl::Layout::kColMajor,
                  "The layout of Global memory and Shared memory tile should "
                  "be row-major.");

    using DType = Shared::DType;

    static_assert(std::is_same_v<typename Global::DType, DType>,
                  "The data type of Shared and Global must be the same.");

    using StoreBase =
        SharedToGlobalBaseTileStorer<Shared, Global, tl::Layout::kColMajor>;

    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    // strides to iterate over each 16x16 `BaseTile` in the shared memory
    static constexpr int kSrcCstride = BaseShape::kCols * Shared::kColStride;
    static constexpr int kDstCstride = BaseShape::kCols * Global::kColStride;

    static constexpr int kRstride = BaseShape::kRows;

    static constexpr int kNumPerAccess = StoreBase::kNumPerAccess;

    DEVICE void operator()(const DType* src, DType* dst) {
        int lane_row = this->lane_row_id() * kNumPerAccess;
        int lane_col = this->lane_col_id();

        int src_lane_offset = src_layout_(lane_row, lane_col);
        int dst_lane_offset = dst_layout_(lane_row, lane_col);

        int src_offset = 0, dst_offset = 0;
        for (int i = 0; i < kRowExec; ++i) {
            for (int j = 0; j < kColExec; ++j) {
                src_offset = i * kRstride + j * kSrcCstride + src_lane_offset;
                dst_offset = i * kRstride + j * kDstCstride + dst_lane_offset;

                this->copy(src + src_offset, dst + dst_offset);
            }
        }
    }

  private:
    typename StoreBase::BaseTileGlobalLayout src_layout_;
    typename StoreBase::BaseTileSharedLayout dst_layout_;
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

    static_assert(kRowExec && kColExec,
                  "Execution count should be greater than 0.");

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

template <typename Shared_, typename WarpLayout_,
          typename Base = warp::CopyBase<WarpLayout_, WarpReuse::kCont>>
struct SharedToGlobalStorer : public Base {
    using Shared = Shared_;
    using DType = Shared::DType;
    using WarpLayout = WarpLayout_;

    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRowExec =
        Base::template row_exec_count<BaseShape, Shared::kRows>();
    static constexpr int kColExec =
        Base::template col_exec_count<BaseShape, Shared::kCols>();

    static_assert(kRowExec && kColExec,
                  "Execution count should be greater than 0.");

    template <typename Global>
    DEVICE void operator()(const Shared& src, Global& dst) {
        const DType* src_ptr = src.data();
        DType* dst_ptr = dst.mutable_data();

        int offset_src = Base::template get_warp_offset<Shared>();
        int offset_dst = Base::template get_warp_offset<Global>();

        using Storer = SharedToGlobalStorerImpl<Shared, Global, kRowExec,
                                                kColExec, Shared::kType>;

        Storer storer;
        storer(src_ptr + offset_src, dst_ptr + offset_dst);
    }
};

}  // namespace tiledcuda::cell::copy
