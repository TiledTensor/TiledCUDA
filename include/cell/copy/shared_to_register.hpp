#pragma once

#include "cell/copy/mod.hpp"
#include "cell/copy/warp.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

using namespace tiledcuda::cell::traits;
using namespace tiledcuda::cell::copy::atom;

namespace detail {

template <typename Shared, const int kRowExec, const int kColExec,
          const tl::Layout kType, CopyInst kCopyInst>
struct SharedToRegLoaderImpl {
    template <typename SrcType, typename DstType>
    DEVICE void operator()(const SrcType* src, DstType& dst);
};

/// @brief partial specialization for row-major shared memory tile.
template <typename Shared, const int kRowExec_, const int kColExec_>
struct SharedToRegLoaderImpl<Shared, kRowExec_, kColExec_, tl::Layout::RowMajor,
                             CopyInst::kLoadMat>
    : public LoadMatBase<typename Shared::DType> {
    using LoadMat = LoadMatBase<typename Shared::DType>;
    using SrcType = Shared::DType;
    using BaseShape = BaseTileShape<SrcType>;

    // strides to iterate over each 16x16 `BaseTile`.
    static constexpr int kTileRstride = BaseShape::kRows * Shared::kRowStride;
    static constexpr int kTileCstride = BaseShape::kCols;

    // row stride and col stride for a thread in a warp
    static constexpr int kStride = LoadMat::kNumPerAccess;
    static constexpr int kLaneRstride = Shared::kRowStride;
    static constexpr int kLaneCstride = kStride;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    template <typename SrcType, typename DstType>
    DEVICE void operator()(const SrcType* src, DstType& dst) {
        int lane_row = this->lane_row_id();
        int lane_col = this->lane_col_id();

        static constexpr int kRowScale =
            BaseShape::kRows / tl::num_rows<typename LoadMat::ThreadLayout>;
        static constexpr int kColScale =
            BaseShape::kCols / (LoadMat::kNumPerAccess *
                                tl::num_cols<typename LoadMat::ThreadLayout>);

        if (thread0()) {
            printf("kRowScale: %d, kColScale: %d\n", kRowScale, kColScale);
        }

        const SrcType* data;
        for (int i = 0; i < kRowExec; ++i) {
            for (int j = 0; j < kColExec; ++j) {
                // 2. advance pointer to the 16x16 `BaseTile` indexed by (i, j).
                data = src + (i * kTileRstride + j * kTileCstride);

                // 3. advance the pointer to data accessed by the current thread
                // inside a 16x16 `BaseTile`.
                data += (lane_row * kLaneRstride + lane_col * kLaneCstride);

                // issue the hardware-backed memory access instruction.
                this->ldmatrix<kRowScale, kColScale>(data,
                                                     dst(i, j).mutable_data());
            }
        }
    }
};

/// @brief partial specialization for column-major shared memory tile.
template <typename Shared, const int kRowExec_, const int kColExec_>
struct SharedToRegLoaderImpl<Shared, kRowExec_, kColExec_, tl::Layout::ColMajor,
                             CopyInst::kLoadMat>
    : public LoadMatBase<typename Shared::DType> {
    using LoadMat = LoadMatBase<typename Shared::DType>;
    using SrcType = Shared::DType;
    using BaseShape = BaseTileShape<SrcType>;

    // strides to iterate over each 16x128-bits `BaseTile`.
    const int kTileRstride = BaseShape::kRows;
    const int kTileCstride = BaseShape::kCols * Shared::kColStride;

    // row stride and col stride for a thread in a warp
    static constexpr int kStride = LoadMat::kNumPerAccess;
    const int kLaneRstride = kStride;
    const int kLaneCstride = Shared::kColStride;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    static constexpr int kRowScale =
        BaseShape::kRows /
        (LoadMat::kNumPerAccess * tl::num_cols<typename LoadMat::ThreadLayout>);
    static constexpr int kColScale =
        BaseShape::kCols / tl::num_rows<typename LoadMat::ThreadLayout>;

    template <typename SrcType, typename DstType>
    DEVICE void operator()(const SrcType* src, DstType& dst) {
        // transpose the lane position if the shared memory is in column-major.
        // 16 threads are mapped to the strided dimension of the data while the
        // 2 threads are mapped to the contiguous dimension of the data.
        int lane_row = this->lane_col_id();
        int lane_col = this->lane_row_id();

        const SrcType* data;
        for (int i = 0; i < kColExec; ++i) {
            for (int j = 0; j < kRowExec; ++j) {
                // 2. advance pointer to the 16x16 `BaseTile` indexed by
                // (i, j).
                data = src + (j * kTileRstride + i * kTileCstride);

                // 3. advance the pointer to data accessed by the current
                // thread inside a 16x128-bits `BaseTile`.
                data += (lane_row * kLaneRstride + lane_col * kLaneCstride);

                // issue the hardware-backed memory access instruction
                this->ldmatrix<kRowScale, kColScale>(data,
                                                     dst(i, j).mutable_data());
            }
        }
    }
};

template <typename Shared, const int kRowExec_, const int kColExec_,
          CopyInst kCopyInst>
struct RegToSharedStorerImpl {
    using DType = typename Shared::DType;

    template <typename SrcType, typename DstType>
    DEVICE void operator()(const SrcType& src, DstType* dst);
};

template <typename Shared, const int kRowExec_, const int kColExec_>
struct RegToSharedStorerImpl<Shared, kRowExec_, kColExec_,
                             CopyInst::kStoreShared32>
    : public StoreMmmaBase<Shared> {
    using StoreBase = StoreMmmaBase<Shared>;
    using SrcType = typename Shared::DType;

    using BaseShape = BaseTileShape<SrcType>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    // strides to iterate over each 16x16 `BaseTile` in the shared memory
    static constexpr int kTileRstride =  // row stride for a `BaseTile`
        Shared::type == tl::Layout::RowMajor
            ? BaseShape::kRows * Shared::kRowStride
            : BaseShape::kRows;
    // FIXME(haruhi): 16 is the magic number for wmma tile. Address these
    // isolated magic numbers.
    static constexpr int kTileCstride =
        Shared::type == tl::Layout::RowMajor ? 16 : 16 * Shared::kColStride;

    // Given the lane position of the current thread, calculate the strides
    // needed to address the data within the 16x128-bit `BaseTile` for the
    // current thread.
    static constexpr int kLaneRstride = Shared::type == tl::Layout::RowMajor
                                            ? Shared::kRowStride
                                            : StoreBase::kStride;
    static constexpr int kLaneCstride = Shared::type == tl::Layout::RowMajor
                                            ? StoreBase::kStride
                                            : Shared::kColStride;

    /// @tparam SrcType the register tile.
    /// @tparam DstType raw data pointer to the shared memory tile.
    template <typename SrcType, typename DstType>
    DEVICE void operator()(const SrcType& src, DstType* dst) {
        DstType* data;
        int lane_row = this->lane_row_id();
        int lane_col = this->lane_col_id();

        for (int i = 0; i < kRowExec; ++i) {
            for (int j = 0; j < kColExec; ++j) {
                // 2. advance pointer to the 16x128-bits `BaseTile` indexed by
                // (i, j).
                data = dst + (i * kTileRstride + j * kTileCstride);
                // 3. advance the pointer to data accessed by the current thread
                // inside a 16x128-bits `Base Tile`.
                data += (lane_row * kLaneRstride + lane_col * kLaneCstride);

                // store the 16x16 `BaseTile` to shared memory
                this->store_base_tile(src(i, j).data(), data);
            }
        }
    };
};

}  // namespace  detail

/// @brief partial specialization for loading data from shared memory to
///        register file using `ldmatrix`.
template <typename Reg_, typename WarpLayout_, const WarpReuse kMode_,
          typename Base = warp::CopyBase<WarpLayout_, kMode_>>
requires RegTileElemType<typename Reg_::DType>
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

        if (thread0()) {
            printf("kRowExec: %d, kColExec: %d\n", kRowExec, kColExec);
        }

        using Loader =
            detail::SharedToRegLoaderImpl<Shared, kRowExec, kColExec,
                                          Shared::type, CopyInst::kLoadMat>;
        Loader loader;
        loader(src_ptr, dst);
    }
};

///@brief partial specialization for 16x16x16 wmma's output, and st.shared.f32
///       to revert the data distrubution into an comphrehensive row-major
///       matrix.
template <typename Reg_, typename WarpLayout_,
          typename Base = warp::CopyBase<WarpLayout_, WarpReuse::Cont>>
requires RegTileElemType<typename Reg_::DType>
struct RegToSharedStorer : public Base {
    using Reg = Reg_;

    // get the elementary data type stored in the register tile.
    using DType = typename Reg::DType::DType;
    using BaseShape = BaseTileShape<DType>;

    using WarpLayout = WarpLayout_;

    template <typename Shared>
    DEVICE void operator()(const Reg& src, Shared& dst) {
        static_assert(std::is_same<typename Shared::DType, DType>::value,
                      "The data type of Shared and Reg must be the same.");
        static_assert(
            (Reg::kNumel * Reg::DType::kNumel * 32 *
             tl::get_numel<WarpLayout>) == Shared::kNumel,
            "The number of elements in Reg and Shared must be the same.");

        DType* dst_ptr = dst.mutable_data();  // pointer for shared memory tile

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

        using Storer = detail::RegToSharedStorerImpl<Shared, kRowExec, kColExec,
                                                     CopyInst::kStoreShared32>;
        Storer storer;
        storer(src, dst_ptr);
    }
};

}  // namespace tiledcuda::cell::copy
