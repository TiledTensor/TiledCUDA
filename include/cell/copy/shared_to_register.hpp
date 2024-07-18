#pragma once

#include "cell/copy/constants.hpp"
#include "cell/copy/warp.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

using namespace tiledcuda::cell::traits;

namespace detail {

struct LoadMatBase {
    using ThreadLdmatrix = tile_layout::ColMajor<16, 2>;

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
    template <typename Element>
    DEVICE void ldmatrix(const Element* src, Element* dst) {
        uint32_t* reg = reinterpret_cast<uint32_t*>(dst);
        uint32_t smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(src));
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
            : "r"(smem_addr));
    }
};

template <typename Shared, const int kRowExec, const int kColExec,
          const tl::Layout kType, CopyInst kCopyInst>
struct SharedToRegLoaderImpl {
    using DType = typename Shared::DType;

    DEVICE void operator()(const DType* src, DType* dst, int row_exec,
                           int col_exec);
};

/// @brief partial specialization for row-major shared memory tile.
template <typename Shared, const int kRowExec_, const int kColExec_>
struct SharedToRegLoaderImpl<Shared, kRowExec_, kColExec_, tl::Layout::RowMajor,
                             CopyInst::LoadMat> : public LoadMatBase {
    using DType = typename Shared::DType;

    // strides to iterate over each 16x128-bits `BaseTile`.
    static constexpr int kTileRstride =
        BaseTileShape<DType>::row * Shared::kRowStride;
    static constexpr int kTileCstride = BaseTileShape<DType>::col;

    // row stride and col stride for a thread in a warp
    static constexpr int kStride = traits::TraitsBase<DType>::kNumPerAccess;
    static constexpr int kLaneRstride = Shared::kRowStride;
    static constexpr int kLaneCstride = kStride;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const DType* src, DType* dst) {
        int lane_row = lane_row_id();
        int lane_col = lane_col_id();

        const DType* data;
        for (int i = 0; i < kRowExec; ++i) {
            for (int j = 0; j < kColExec; ++j) {
                // 2. advance pointer to the 16x128-bits `BaseTile` indexed by
                // (i, j).
                data = src + (i * kTileRstride + j * kTileCstride);

                // 3. advance the pointer to data accessed by the current thread
                // inside a 16x128-bits `BaseTile`.
                data += (lane_row * kLaneRstride + lane_col * kLaneCstride);

                // issue the hardware-backed memory access instruction.
                LoadMatBase::template ldmatrix<DType>(data, dst);

                // advance the pointer to store the next 16x128-bits `BaseTile`.
                dst += BaseTileShape<DType>::elem_per_thread;
            }
        }
    }
};

/// @brief partial specialization for column-major shared memory tile.
template <typename Shared, const int kRowExec_, const int kColExec_>
struct SharedToRegLoaderImpl<Shared, kRowExec_, kColExec_, tl::Layout::ColMajor,
                             CopyInst::LoadMat> : public LoadMatBase {
    using DType = typename Shared::DType;

    // strides to iterate over each 16x128-bits `BaseTile`.
    const int kTileRstride = BaseTileShape<DType>::row;
    const int kTileCstride = BaseTileShape<DType>::col * Shared::kColStride;

    // row stride and col stride for a thread in a warp
    static constexpr int kStride = traits::TraitsBase<DType>::kNumPerAccess;
    const int kLaneRstride = kStride;
    const int kLaneCstride = Shared::kColStride;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const DType* src, DType* dst) {
        // transpose the lane position if the shared memory is in column-major.
        // 16 threads are mapped to the strided dimension of the data while the
        // 2 threads are mapped to the contiguous dimension of the data.
        int lane_row = lane_col_id();
        int lane_col = lane_row_id();

        const DType* data;
        for (int i = 0; i < kColExec; ++i) {
            for (int j = 0; j < kRowExec; ++j) {
                // 2. advance pointer to the 16x128-bits `BaseTile` indexed by
                // (i, j).
                data = src + (j * kTileRstride + i * kTileCstride);

                // 3. advance the pointer to data accessed by the current
                // thread inside a 16x128-bits `BaseTile`.
                data += (lane_row * kLaneRstride + lane_col * kLaneCstride);

                // issue the hardware-backed memory access instruction
                LoadMatBase::template ldmatrix<DType>(data, dst);

                // advance the pointer to store the next 16x128-bits `BaseTile`.
                dst += BaseTileShape<DType>::elem_per_thread;
            }
        }
    }
};

template <typename Shared, const int kRowExec_, const int kColExec_,
          CopyInst kCopyInst>
struct RegToSharedStorerImpl {
    using DType = typename Shared::DType;

    DEVICE void operator()(const DType* src, DType* dst);
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
    // unit of a TCU instruction. Regardless of the output precision, a single
    // execution of WMMA stores 2 elements in each thread's local register.
    static constexpr int kStride = 2;
    static constexpr int kRstride =
        Shared::type == tl::Layout::RowMajor
            ? tl::num_rows<ThreadWmma> * Shared::kRowStride
            : tl::num_rows<ThreadWmma>;
    static constexpr int kCstride =
        Shared::type == tl::Layout::RowMajor
            ? kStride * tl::num_cols<ThreadWmma>
            : kStride * tl::num_cols<ThreadWmma> * Shared::kColStride;

    // strides to iterate over each 16x128-bits `BaseTile` in the shared memory
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

    DEVICE void operator()(const DType* src, DType* dst) {
        DType* data;
        int lane_row = lane_row_id();
        int lane_col = lane_col_id();

        for (int i = 0; i < kRowExec; ++i) {
            for (int j = 0; j < kColExec; ++j) {
                // 2. advance pointer to the 16x128-bits `BaseTile` indexed by
                // (i, j).
                data = dst + (i * kTileRstride + j * kTileCstride);
                // 3. advance the pointer to data accessed by the current thread
                // inside a 16x128-bits `Base Tile`.
                data += (lane_row * kLaneRstride + lane_col * kLaneCstride);

                // store the 16x128-bits `BaseTile` to shared memory
                store_base_tile(src, data);

                // FIXME(haruhi): The value 8 represents the number of elements
                // stored in each thread's local register during a single
                // execution of the WMMA instruction. Replace this magic number
                // with a well-managed constant.
                src += 8;
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
    using DType = typename Reg::DType;

    using WarpLayout = WarpLayout_;
    static constexpr WarpReuse kMode = kMode_;

    template <typename Shared>
    DEVICE void operator()(const Shared& src, Reg& dst) {
        static_assert(std::is_same<typename Shared::DType, DType>::value,
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
            Base::template row_exec_count<DType, Shared::kRows>();
        static constexpr int kColExec =
            Base::template col_exec_count<DType, Shared::kCols>();

        using Loader =
            detail::SharedToRegLoaderImpl<Shared, kRowExec, kColExec,
                                          Shared::type, CopyInst::LoadMat>;
        Loader loader;
        loader(src_ptr, dst.mutable_data());
    }
};

///@brief partial specialization for wmma 16x16x16, and LDSM32
template <typename Reg_, typename WarpLayout_,
          typename Base = warp::CopyBase<WarpLayout_, WarpReuse::Cont>>
struct RegToSharedStorer : public Base {
    using Reg = Reg_;
    using WarpLayout = WarpLayout_;
    using DType = typename Reg::DType;

    template <typename Shared>
    DEVICE void operator()(const Reg& src, Shared& dst) {
        static_assert(std::is_same<typename Shared::DType, DType>::value,
                      "The data type of Shared and Reg must be the same.");
        static_assert(
            (Reg::kNumel * 32 * tl::get_numel<WarpLayout>) == Shared::kNumel,
            "The number of elements in Reg and Shared must be the same.");

        DType* dst_ptr = dst.mutable_data();  // pointer for shared memory tile

        static constexpr int kRowExec =
            Base::template row_exec_count<DType, Shared::kRows>();
        static constexpr int kColExec =
            Base::template col_exec_count<DType, Shared::kCols>();

        // 1. advance the pointer to input data to the current warp according to
        // warp reuse mode. During the store process, threads do not write to
        // the same shared memory location, thus the warp reuse mode is set to
        // `Cont`.
        int offset = Base::template get_warp_offset<Shared>();
        dst_ptr += offset;

        using Storer = detail::RegToSharedStorerImpl<Shared, kRowExec, kColExec,
                                                     CopyInst::LoadS32>;
        Storer storer;
        storer(src.data(), dst_ptr);
    }
};

}  // namespace tiledcuda::cell::copy
