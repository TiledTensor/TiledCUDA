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
struct SharedToRegLoaderImpl;

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

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    using BaseTilesLayout =
        tl::MatrixLayout<kRowExec, kColExec,
                         BaseShape::kRows * Shared::kRowStride,
                         BaseShape::kCols>;

    using BaseTileSharedLayout = tl::SharedLayoutWrapper<Shared>::Layout;

    DEVICE SharedToRegLoaderImpl()
        : base_tiles_(BaseTilesLayout{}),
          in_base_tile_(BaseTileSharedLayout{}) {}

    DEVICE void operator()(const DType* src, Reg& dst) {
        int lane_row = this->lane_row_id();
        int lane_col = this->lane_col_id() * LoadMat::kNumPerAccess;

        int lane_offset = in_base_tile_(lane_row, lane_col);
        int offset = 0;

#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                // advance pointer to the 16x16 `BaseTile` indexed by(i, j).
                offset = base_tiles_(i, j) + lane_offset;
                // issue the hardware-backed memory access instruction.
                this->ldmatrix(&src[offset], dst(i, j).mutable_data());
            }
        }
    }

  private:
    BaseTilesLayout base_tiles_;
    BaseTileSharedLayout in_base_tile_;
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

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    using BaseTilesLayout =
        tl::MatrixLayout<kRowExec, kColExec, BaseShape::kRows,
                         BaseShape::kCols * Shared::kColStride>;

    using BaseTileSharedLayout = tl::SharedLayoutWrapper<Shared>::Layout;

    DEVICE SharedToRegLoaderImpl()
        : base_tiles_(BaseTilesLayout{}),
          in_base_tile_(BaseTileSharedLayout{}) {}

    DEVICE void operator()(const DType* src, Reg& dst) {
        // transpose the lane position if the shared memory is in column-major.
        // 16 threads are mapped to the strided dimension of the data while the
        // 2 threads are mapped to the contiguous dimension of the data.
        int lane_row = this->lane_col_id() * LoadMat::kNumPerAccess;
        int lane_col = this->lane_row_id();

        int lane_offset = in_base_tile_(lane_row, lane_col);
        int offset = 0;

        for (int i = 0; i < kColExec; ++i) {
#pragma unroll
            for (int j = 0; j < kRowExec; ++j) {
                offset = base_tiles_(j, i) + lane_offset;

                // issue the hardware-backed memory access instruction
                this->ldmatrix(&src[offset], dst(j, i).mutable_data());
            }
        }
    }

  private:
    BaseTilesLayout base_tiles_;
    BaseTileSharedLayout in_base_tile_;
};

template <typename Shared, typename Reg_, const int kRowExec_,
          const int kColExec_, const tl::Layout kType>
struct RegToSharedStorerImpl;

template <typename Shared, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct RegToSharedStorerImpl<Shared, Reg_, kRowExec_, kColExec_,
                             tl::Layout::kRowMajor> {
    using Reg = Reg_;
    using DType = typename Shared::DType;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const Reg& src, DType* dst) {
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                storer_(src(i, j).data(), dst + base_tiles_(i, j));
            }
        }
    }

  private:
    using BaseShape = BaseTileShape<DType>;
    using BaseTilesLayout =
        tl::MatrixLayout<kRowExec, kColExec,
                         BaseShape::kRows * Shared::kRowStride,
                         BaseShape::kCols>;
    BaseTilesLayout base_tiles_;

    using Storer = BaseTileStorer<Shared, Shared::kType, sizeof(DType) * 8>;
    Storer storer_;
};

template <typename Shared, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct RegToSharedStorerImpl<Shared, Reg_, kRowExec_, kColExec_,
                             tl::Layout::kColMajor> {
    using Reg = Reg_;
    using DType = typename Shared::DType;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const Reg& src, DType* dst) {
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                storer_(src(i, j).data(), dst + base_tiles_(i, j));
            }
        }
    }

  private:
    using BaseShape = BaseTileShape<DType>;
    using BaseTilesLayout =
        tl::MatrixLayout<kRowExec, kColExec, BaseShape::kRows,
                         BaseShape::kCols * Shared::kColStride>;
    BaseTilesLayout base_tiles_;

    using Storer = BaseTileStorer<Shared, Shared::kType, sizeof(DType) * 8>;
    Storer storer_;
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

        // how many times a `BaseTile` is executed along the row and column
        // direction.
        static constexpr int kRowExec =
            Base::template row_exec_count<BaseShape, Shared::kRows>();
        static constexpr int kColExec =
            Base::template col_exec_count<BaseShape, Shared::kCols>();

        // advance the pointer to input data to the current warp according to
        // warp reuse mode.
        const DType* src_ptr = src.data();
        int offset = Base::template get_warp_offset<Shared>();

        using Loader =
            detail::SharedToRegLoaderImpl<Shared, Reg, kRowExec, kColExec,
                                          Shared::kType, CopyInst::kLoadMat>;
        Loader loader;
        loader(src_ptr + offset, dst);
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
        static_assert(
            Shared::kType == Reg::kType,
            "The layout of Shared and Register tile must be the same.");

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

        using Storer = detail::RegToSharedStorerImpl<Shared, Reg, kRowExec,
                                                     kColExec, Shared::kType>;
        Storer storer;
        storer(src, dst_ptr + offset);
    }
};

}  // namespace tiledcuda::cell::copy
