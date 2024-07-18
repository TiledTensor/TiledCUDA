#pragma once

#include "cell/copy/constants.hpp"
#include "cell/copy/warp.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

using namespace tiledcuda::cell::traits;

namespace detail {

template <typename Element>
struct LoadMatBase {
    using DType = Element;

    using ThreadLdmatrix = tile_layout::ColMajor<16, 2>;

    static constexpr int kAccessInBits = 128;  // 128 bits
    static constexpr int kElmentBits = sizeof(DType) * 8;
    static constexpr int kNumPerAccess = kAccessInBits / kElmentBits;

    // @brief returns the lane row of the current thread within a warp.
    //        For an example, in ldmatrix, threads in a warp are arranged as
    //        follows (a 16 x 2 column-major): |  | 0 |  1|
    //        |--|---|---|
    //        |0 | 0 | 16|
    //        |1 | 2 | 17|
    //        |2 | 4 | 18|
    //        |  |...|...|
    //        |15| 15| 31|
    //        if threadIdx.x is 43, then its lane row is 8, lane col is 0.
    DEVICE int lane_row_id() {
        int lane_id = threadIdx.x % warpSize;
        return lane_id % tl::num_rows<ThreadLdmatrix>;
    }

    // @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_col_id() {
        int lane_id = threadIdx.x % warpSize;
        return lane_id / tl::num_rows<ThreadLdmatrix>;
    }

    /// @brief a thin wrapper for executing a single ldmatrix instruction.
    DEVICE void ldmatrix(const DType* src, DType* dst) {
        uint32_t* reg = reinterpret_cast<uint32_t*>(dst);
        uint32_t smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(src));
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, "
            "[%4];\n"
            : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
            : "r"(smem_addr));
    }
};

template <typename Shared, const int kRowExec, const int kColExec,
          const tl::Layout kType, CopyInst kCopyInst>
struct SharedToRegLoaderImpl {
    template <typename SrcType, typename DstType>
    DEVICE void operator()(const SrcType* src, DstType* dst);
};

/// @brief partial specialization for row-major shared memory tile.
template <typename Shared, const int kRowExec_, const int kColExec_>
struct SharedToRegLoaderImpl<Shared, kRowExec_, kColExec_, tl::Layout::RowMajor,
                             CopyInst::LoadMat>
    : public LoadMatBase<typename Shared::DType> {
    using SrcType = Shared::DType;
    using BaseShape = BaseTileShape<SrcType>;

    // strides to iterate over each 16x16 `BaseTile`.
    static constexpr int kTileRstride = BaseShape::kRows * Shared::kRowStride;
    static constexpr int kTileCstride = BaseShape::kCols;

    // row stride and col stride for a thread in a warp
    static constexpr int kStride = LoadMatBase<SrcType>::kNumPerAccess;
    static constexpr int kLaneRstride = Shared::kRowStride;
    static constexpr int kLaneCstride = kStride;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    template <typename SrcType, typename DstType>
    DEVICE void operator()(const SrcType* src, DstType& dst) {
        int lane_row = this->lane_row_id();
        int lane_col = this->lane_col_id();

        const SrcType* data;
        for (int i = 0; i < kRowExec; ++i) {
            for (int j = 0; j < kColExec; ++j) {
                // 2. advance pointer to the 16x16 `BaseTile` indexed by (i,
                // j).
                data = src + (i * kTileRstride + j * kTileCstride);

                // 3. advance the pointer to data accessed by the current
                // thread inside a 16x16 `BaseTile`.
                data += (lane_row * kLaneRstride + lane_col * kLaneCstride);

                // issue the hardware-backed memory access instruction.
                this->ldmatrix(data, dst(i, j).mutable_data());
            }
        }
    }
};

/// @brief partial specialization for column-major shared memory tile.
template <typename Shared, const int kRowExec_, const int kColExec_>
struct SharedToRegLoaderImpl<Shared, kRowExec_, kColExec_, tl::Layout::ColMajor,
                             CopyInst::LoadMat>
    : public LoadMatBase<typename Shared::DType> {
    using SrcType = Shared::DType;
    using BaseShape = BaseTileShape<SrcType>;

    // strides to iterate over each 16x128-bits `BaseTile`.
    const int kTileRstride = BaseShape::kRows;
    const int kTileCstride = BaseShape::kCols * Shared::kColStride;

    // row stride and col stride for a thread in a warp
    static constexpr int kStride = LoadMatBase<SrcType>::kNumPerAccess;
    const int kLaneRstride = kStride;
    const int kLaneCstride = Shared::kColStride;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    template <typename SrcType, typename DstType>
    DEVICE void operator()(const SrcType* src, DstType* dst) {
        // transpose the lane position if the shared memory is in
        // column-major. 16 threads are mapped to the strided dimension of
        // the data while the 2 threads are mapped to the contiguous
        // dimension of the data.
        int lane_row = this->lane_col_id();
        int lane_col = this->lane_row_id();

        const SrcType* data;
        for (int i = 0; i < kColExec; ++i) {
            for (int j = 0; j < kRowExec; ++j) {
                // 2. advance pointer to the 16x16 `BaseTile` indexed by (i,
                // j).
                data = src + (j * kTileRstride + i * kTileCstride);

                // 3. advance the pointer to data accessed by the current
                // thread inside a 16x128-bits `BaseTile`.
                data += (lane_row * kLaneRstride + lane_col * kLaneCstride);

                // issue the hardware-backed memory access instruction
                this->ldmatrix(data, dst(i, j).mutable_data());
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
struct RegToSharedStorerImpl<Shared, kRowExec_, kColExec_, CopyInst::LoadS32> {
    using DType = typename Shared::DType;
    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    using ThreadWmma = tile_layout::RowMajor<8, 4>;

    // FIXME(haruhi): Try to find a better way to organize these
    // hardware-dependent magic numbers.
    // The number 2 is interpreted as follows: 8x8 block is a fundamental
    // unit of a TCU instruction. Regardless of the output precision, a
    // single execution of WMMA stores 2 elements in each thread's local
    // register.
    static constexpr int kStride = 2;
    static constexpr int kRstride =
        Shared::type == tl::Layout::RowMajor
            ? tl::num_rows<ThreadWmma> * Shared::kRowStride
            : tl::num_rows<ThreadWmma>;
    static constexpr int kCstride =
        Shared::type == tl::Layout::RowMajor
            ? kStride * tl::num_cols<ThreadWmma>
            : kStride * tl::num_cols<ThreadWmma> * Shared::kColStride;

    // strides to iterate over each 16x16 `BaseTile` in the shared memory
    static constexpr int kTileRstride =  // row stride for a `BaseTile`
        Shared::type == tl::Layout::RowMajor
            ? BaseTileShape<DType>::row * Shared::kRowStride
            : BaseTileShape<DType>::row;

    // FIXME(haruhi): 16 is the magic number for wmma tile. Address these
    // isolated magic numbers.
    static constexpr int kTileCstride =
        Shared::type == tl::Layout::RowMajor ? 16 : 16 * Shared::kColStride;

    // Given the lane position of the current thread, calculate the strides
    // needed to address the data within the 16x128-bit `BaseTile` for the
    // current thread.
    static constexpr int kLaneRstride =
        Shared::type == tl::Layout::RowMajor ? Shared::kRowStride : kStride;
    static constexpr int kLaneCstride =
        Shared::type == tl::Layout::RowMajor ? kStride : Shared::kColStride;

    /// @tparam SrcType the register tile.
    /// @tparam DstType raw data pointer to the shared memory tile.
    template <typename SrcType, typename DstType>
    DEVICE void operator()(const SrcType& src, DstType* dst) {
        DstType* data;
        int lane_row = this->lane_row_id();
        int lane_col = this->lane_col_id();

        for (int i = 0; i < kRowExec; ++i) {
            for (int j = 0; j < kColExec; ++j) {
                // 2. advance pointer to the 16x128-bits `BaseTile` indexed
                // by (i, j).
                data = dst + (i * kTileRstride + j * kTileCstride);
                // 3. advance the pointer to data accessed by the current
                // thread inside a 16x128-bits `Base Tile`.
                data += (lane_row * kLaneRstride + lane_col * kLaneCstride);

                // store the 16x16 `BaseTile` to shared memory
                store_base_tile(src(i, j).data(), data);
            }
        }
    };

  private:
    // TODO: Improve this is a naive implementation that does not use
    // vectorized access in future. To use vectorized access, we need to
    // know the return type of WMMA.
    DEVICE void store_base_tile(const DType* src, DType* dst) {
        int src_offset, dst_offset;
        // TODO(haruhi): comments the magic number
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                src_offset = i * 4 + j * 2;
                // Be cautious, j and i are swapped here. DONOT change this
                dst_offset = j * kRstride + i * kCstride;

                dst[dst_offset] = src[src_offset];
                dst[dst_offset + 1] = src[src_offset + 1];
            }
        }
    }

    DEVICE int lane_row_id() {
        return (threadIdx.x % warpSize) / tl::num_cols<ThreadWmma>;
    }

    DEVICE int lane_col_id() {
        return (threadIdx.x % warpSize) % tl::num_cols<ThreadWmma>;
    }
};

}  // namespace  detail

/// @brief partial specialization for loading data from shared memory to
///        register file using `ldmatrix`.
template <typename Reg_, typename WarpLayout_, const WarpReuse kMode_,
          typename Base = warp::CopyBase<WarpLayout_, kMode_>>
struct SharedToRegLoader : public Base {
    using Reg = Reg_;

    static_assert(
        std::is_same_v<typename Reg::DType, BaseHalfTileRowMajor> ||
            std::is_same_v<typename Reg::DType, BaseHalfTileColMajor> ||
            std::is_same_v<typename Reg::DType, BaseFloatTileRowMajor> ||
            std::is_same_v<typename Reg::DType, BaseFloatTileColMajor>,
        "unsupported data type for register file.");

    using DType = typename Reg::DType::DType;  // the element data type
    using BaseTile = BaseTileShape<DType>;

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

        // 1. advance the pointer to input data to the current warp
        // according to warp reuse mode.
        const DType* src_ptr = src.data();
        src_ptr += Base::template get_warp_offset<Shared>();

        // how many times a `BaseTile` is executed along the row and column
        // direction.
        static constexpr int kRowExec =
            Base::template row_exec_count<BaseTile, Shared::kRows>();
        static constexpr int kColExec =
            Base::template col_exec_count<BaseTile, Shared::kCols>();

        if (thread0()) {
            printf("kRowExec: %d, kColExec: %d\n", kRowExec, kColExec);
        }

        using Loader =
            detail::SharedToRegLoaderImpl<Shared, kRowExec, kColExec,
                                          Shared::type, CopyInst::LoadMat>;
        Loader loader;
        loader(src_ptr, dst);
    }
};

///@brief partial specialization for wmma 16x16x16, and LDSM32
template <typename Reg_, typename WarpLayout_,
          typename Base = warp::CopyBase<WarpLayout_, WarpReuse::Cont>>
struct RegToSharedStorer : public Base {
    using Reg = Reg_;

    static_assert(
        std::is_same_v<typename Reg::DType, BaseHalfTileRowMajor> ||
            std::is_same_v<typename Reg::DType, BaseHalfTileColMajor> ||
            std::is_same_v<typename Reg::DType, BaseFloatTileRowMajor> ||
            std::is_same_v<typename Reg::DType, BaseFloatTileColMajor>,
        "unsupported data type for register file.");

    using DType = typename Reg::DType::DType;  // the elementary data type
    using BaseTile = BaseTileShape<DType>;

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
            Base::template row_exec_count<BaseTile, Shared::kRows>();
        static constexpr int kColExec =
            Base::template col_exec_count<BaseTile, Shared::kCols>();

        // 1. advance the pointer to input data to the current warp
        // according to warp reuse mode. During the store process, threads
        // do not write to the same shared memory location, thus the warp
        // reuse mode is set to `Cont`.
        int offset = Base::template get_warp_offset<Shared>();
        dst_ptr += offset;

        using Storer = detail::RegToSharedStorerImpl<Shared, kRowExec, kColExec,
                                                     CopyInst::LoadS32>;
        Storer storer;
        storer(src, dst_ptr);
    }
};

}  // namespace tiledcuda::cell::copy
