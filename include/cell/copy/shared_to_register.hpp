#pragma once

#include "cell/copy/mod.hpp"
#include "cell/copy/warp.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

using namespace traits;
using namespace atom;
namespace tl = tile_layout;

namespace detail {

template <typename Shared, typename Reg_, const int kRowExec,
          const int kColExec, const tl::Layout kType, CopyInst kCopyInst>
struct SharedToRegLoaderImpl {
    using DType = typename Shared::DType;
    using Reg = Reg_;

    DEVICE void operator()(const DType* src, Reg& dst);
};

/// @brief partial specialization for row-major shared memory tile.
template <typename Shared, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct SharedToRegLoaderImpl<Shared, Reg_, kRowExec_, kColExec_,
                             tl::Layout::kRowMajor, CopyInst::kLoadMat>
    : public LoadMatBase<typename Shared::DType> {
    using LoadMat = LoadMatBase<typename Shared::DType>;
    using DType = Shared::DType;
    using Reg = Reg_;
    using BaseShape = BaseTileShape<DType>;

    // strides to iterate over each 16x16 `BaseTile`.
    // static constexpr int kTileRstride = BaseShape::kRows *
    // Shared::kRowStride; static constexpr int kTileCstride = BaseShape::kCols;

    // row stride and col stride for a thread in a warp
    // static constexpr int kStride = LoadMat::kNumPerAccess;
    // static constexpr int kLaneRstride = Shared::kRowStride;
    // static constexpr int kLaneCstride = kStride;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE SharedToRegLoaderImpl() : layout_(typename Shared::Layout{}) {}

    DEVICE void operator()(const DType* src, Reg& dst) {
        int lane_row = this->lane_row_id();
        int lane_col = this->lane_col_id();

        // const DType* data;
        // const __half* data_ptr = reinterpret_cast<const __half*>(src);
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                int x = i * 16 + lane_row;
                int y = j * 16 + lane_col * 8;
                int offset = layout_(x, y);

                // printf(
                //     "[%d,%d]-%d-1: %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f,
                //     "
                //     "%.0f\n",
                //     i, j, threadIdx.x, __half2float(data_ptr[offset]),
                //     __half2float(data_ptr[offset + 1]),
                //     __half2float(data_ptr[offset + 2]),
                //     __half2float(data_ptr[offset + 3]),
                //     __half2float(data_ptr[offset + 4]),
                //     __half2float(data_ptr[offset + 5]),
                //     __half2float(data_ptr[offset + 6]),
                //     __half2float(data_ptr[offset + 7]));
                this->ldmatrix(src + offset, dst(i, j).mutable_data());
                // printf("%d-2\n", threadIdx.x);

                // 2. advance pointer to the 16x16 `BaseTile` indexed by (i, j).
                // data = src + (i * kTileRstride + j * kTileCstride);

                // // 3. advance the pointer to data accessed by the current
                // // thread inside a 16x16 `BaseTile`.
                // data += (lane_row * kLaneRstride + lane_col * kLaneCstride);

                // // issue the hardware-backed memory access instruction.
                // this->ldmatrix(data, dst(i, j).mutable_data());

                // if (thread(10)) {
                //     // int x = i * kTileRstride + lane_row * kLaneRstride;
                //     // int y = j * kTileCstride + lane_col * kLaneCstride;
                //     int x = i * 16 + lane_row;
                //     int y = j * 16 + lane_col;
                //     printf(
                //         "\niter-(%d, %d): lane pos = [%d, %d], pos = [%d, "
                //         "%d]\n",
                //         i, j, lane_row, lane_col, x, y);

                //     const __half* data_ptr =
                //         reinterpret_cast<const __half*>(src);

                //     int offset = layout_(x, y);
                //     printf("use layout:\n");
                //     for (int i = 0; i < 8; ++i) {
                //         printf("%.0f,", __half2float(data_ptr[offset + i]));
                //     }
                // }
            }
        }
    }

  private:
    typename Shared::Layout layout_;
};

/// @brief partial specialization for column-major shared memory tile.
template <typename Shared, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct SharedToRegLoaderImpl<Shared, Reg_, kRowExec_, kColExec_,
                             tl::Layout::kColMajor, CopyInst::kLoadMat>
    : public LoadMatBase<typename Shared::DType> {
    using LoadMat = LoadMatBase<typename Shared::DType>;
    using DType = Shared::DType;
    using Reg = Reg_;
    using BaseShape = BaseTileShape<DType>;

    // strides to iterate over each 16x128-bits `BaseTile`.
    const int kTileRstride = BaseShape::kRows;
    const int kTileCstride = BaseShape::kCols * Shared::kColStride;

    // row stride and col stride for a thread in a warp
    static constexpr int kStride = LoadMat::kNumPerAccess;
    const int kLaneRstride = kStride;
    const int kLaneCstride = Shared::kColStride;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const DType* src, Reg& dst) {
        // transpose the lane position if the shared memory is in column-major.
        // 16 threads are mapped to the strided dimension of the data while the
        // 2 threads are mapped to the contiguous dimension of the data.
        int lane_row = this->lane_col_id();
        int lane_col = this->lane_row_id();

        const DType* data;
#pragma unroll
        for (int i = 0; i < kColExec; ++i) {
#pragma unroll
            for (int j = 0; j < kRowExec; ++j) {
                // 2. advance pointer to the 16x16 `BaseTile` indexed by
                // (i, j).
                data = src + (j * kTileRstride + i * kTileCstride);

                // 3. advance the pointer to data accessed by the current
                // thread inside a 16x128-bits `BaseTile`.
                data += (lane_row * kLaneRstride + lane_col * kLaneCstride);

                // issue the hardware-backed memory access instruction
                this->ldmatrix(data, dst(j, i).mutable_data());
            }
        }
    }
};

template <typename Shared, typename Reg_, const int kRowExec_,
          const int kColExec_, CopyInst kCopyInst>
struct RegToSharedStorerImpl {
    using DType = typename Shared::DType;
    using Reg = Reg_;

    DEVICE void operator()(const Reg& src, DType* dst);
};

template <typename Shared, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct RegToSharedStorerImpl<Shared, Reg_, kRowExec_, kColExec_,
                             CopyInst::kStoreShared32>
    : public StoreMmmaBase<Shared> {
    using Base = StoreMmmaBase<Shared>;

    using Reg = Reg_;
    using DType = typename Shared::DType;
    using BaseShape = BaseTileShape<DType>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    // strides to iterate over each 16x16 `BaseTile` in the shared memory
    static constexpr int kTileRstride =  // row stride for a `BaseTile`
        Shared::kType == tl::Layout::kRowMajor
            ? BaseShape::kRows * Shared::kRowStride
            : BaseShape::kRows;
    static constexpr int kTileCstride =
        Shared::kType == tl::Layout::kRowMajor
            ? BaseShape::kCols
            : BaseShape::kCols * Shared::kColStride;

    // Given the lane position of the current thread, calculate the strides
    // needed to address the data within the 16x16 `BaseTile` for the current
    // thread. These strides corresponds to the thread layout and specific
    // instruction used.
    static constexpr int kLaneRstride = Shared::kType == tl::Layout::kRowMajor
                                            ? Shared::kRowStride
                                            : Base::kElemPerSeg;
    static constexpr int kLaneCstride = Shared::kType == tl::Layout::kRowMajor
                                            ? Base::kElemPerSeg
                                            : Shared::kColStride;

    DEVICE void operator()(const Reg& src, DType* dst) {
        DType* data;
        int lane_row = this->lane_row_id();
        int lane_col = this->lane_col_id();

#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                // 2. advance pointer to the 16x128-bits `BaseTile` indexed by
                // (i, j).
                data = dst + (i * kTileRstride + j * kTileCstride);
                // 3. advance the pointer to data accessed by the current thread
                // inside a 16x16 `BaseTile`.
                data += (lane_row * kLaneRstride + lane_col * kLaneCstride);

                // store the 16x16 `BaseTile` to shared memory
                this->store_base_tile(src(i, j), data);
            }
        }
    };
};

}  // namespace  detail

/// @brief partial specialization for loading data from shared memory to
///        register file using `ldmatrix`.
template <typename Reg_, typename WarpLayout_, const WarpReuse kMode_,
          typename Base = warp::CopyBase<WarpLayout_, kMode_>>
struct SharedToRegLoader : public Base {
    using Reg = Reg_;
    using DType = typename Reg::DType::DType;  // the element data type
    using BaseShape = BaseTileShape<DType>;

    using WarpLayout = WarpLayout_;
    static constexpr WarpReuse kMode = kMode_;

    template <typename Shared>
    DEVICE void operator()(const Shared& src, Reg& dst) {
        static_assert(std::is_same_v<typename Shared::DType, DType>,
                      "The data type of Shared and Reg must be the same.");
        static_assert(Shared::kRows % tl::num_rows<WarpLayout> == 0,
                      "The current implementation requires Shared::kRows must "
                      "be divisible by tl::num_rows<WarpLayout>");
        static_assert(Shared::kCols % tl::num_cols<WarpLayout> == 0,
                      "The current implementation requires Shared::kCols must "
                      "be divisible by tl::num_cols<WarpLayout>");

        // 1. advance the pointer to input data to the current warp according to
        // warp reuse mode.
        const DType* src_ptr = src.data();
        src_ptr += Base::template get_warp_offset<Shared>();

        // how many times a `BaseTile` is executed along the row and column
        // direction.
        static constexpr int kRowExec =
            Base::template row_exec_count<BaseShape, Shared::kRows>();
        static constexpr int kColExec =
            Base::template col_exec_count<BaseShape, Shared::kCols>();

        using Loader =
            detail::SharedToRegLoaderImpl<Shared, Reg, kRowExec, kColExec,
                                          Shared::kType, CopyInst::kLoadMat>;
        Loader loader;
        loader(src_ptr, dst);
    }
};

/// @brief partial specialization for 16x16x16 wmma's output, and st.shared.f32
///        to revert the data distrubution into an comphrehensive row-major
///        matrix.
template <typename Reg_, typename WarpLayout_,
          typename Base = warp::CopyBase<WarpLayout_, WarpReuse::kCont>>
struct RegToSharedStorer : public Base {
    using Reg = Reg_;
    // elementary data type stored in the register tile.
    using DType = typename Reg::DType::DType;
    using BaseShape = BaseTileShape<DType>;

    using WarpLayout = WarpLayout_;

    /// @brief Store the WMMA output register tile to shared memory. The source
    ///        is the current thread's local register tile, and the destination
    ///        is shared memory.
    template <typename Shared>
    DEVICE void operator()(const Reg& src, Shared& dst) {
        static_assert(std::is_same_v<typename Shared::DType, DType>,
                      "The element data type of Shared and Register tile must "
                      "be the same.");
        static_assert((Reg::kNumel * Reg::DType::kNumel * 32 /*warp size*/ *
                       tl::get_numel<WarpLayout>) == Shared::kNumel,
                      "The number of elements held in the local register file "
                      "by all threads in the CTA must be the same as the "
                      "number held in the shared memory tile.");

        DType* dst_ptr = dst.mutable_data();  // pointer for shared memory tile

        // how many times the 16x16 `BaseTile` is executed along the row and
        // column direction.
        static constexpr int kRowExec =
            Base::template row_exec_count<BaseShape, Shared::kRows>();
        static constexpr int kColExec =
            Base::template col_exec_count<BaseShape, Shared::kCols>();

        // 1. advance the pointer to input data to the current warp according to
        // warp reuse mode. During the store process, threads do not write to
        // the same shared memory location, thus the warp reuse mode is set to
        // `Cont`.
        int offset = Base::template get_warp_offset<Shared>();
        dst_ptr += offset;

        using Storer =
            detail::RegToSharedStorerImpl<Shared, Reg, kRowExec, kColExec,
                                          CopyInst::kStoreShared32>;
        Storer storer;
        storer(src, dst_ptr);
    }
};

}  // namespace tiledcuda::cell::copy
