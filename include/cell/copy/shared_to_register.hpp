#pragma once

#include "cell/copy/constants.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

using namespace tiledcuda::cell::traits;

namespace detail {

template <typename WarpLayout_, const WarpReuse kMode_>
struct CopyBase {
    static constexpr WarpReuse kMode = kMode_;
    using WarpLayout = WarpLayout_;

    // @brief This function returns the offset to the start position of the
    //        current warp in the shared memory according to the warp reuse
    //        mode.
    template <typename Shared>
    DEVICE int get_warp_offset() {
        // Tile shape for a single warp
        constexpr static int warp_shape_row =
            Shared::kRows / tl::num_rows<WarpLayout>;
        constexpr static int warp_shape_col =
            Shared::kCols / tl::num_cols<WarpLayout>;

        constexpr static int warp_rstride =
            Shared::type == tl::Layout::RowMajor
                ? Shared::kRowStride * warp_shape_row
                : warp_shape_row;
        constexpr static int warp_cstride =
            Shared::type == tl::Layout::RowMajor
                ? warp_shape_col
                : Shared::kColStride * warp_shape_col;

        switch (kMode) {
            case WarpReuse::Cont: {
                // In 'Cont' mode, all warps evenly divide the large data tiles
                // and load the data without repeated loads.
                int warp_row = warp_row_id();
                int warp_col = warp_col_id();
                return warp_row * warp_rstride + warp_col * warp_cstride;
            }
            case WarpReuse::RowReuseCont: {
                // In the `RowReuseCont` mode, warps in a same row repeatedly
                // load the same data, and each warp loads a continuous chunks
                // of data.
                int warp_row = warp_row_id();
                return warp_row * warp_rstride;
            }
            case WarpReuse::ColReuseCont: {
                // In the `ColReuseCont` mode, warps in a same column repeatedly
                // load the same data, and each warp loads a continuous chunks
                // of data.
                int warp_col = warp_col_id();
                return warp_col * warp_cstride;
            }
            default:
                assert(false && "Not implemented yet.");
                return -1;
        }
    }

    // @brief This function returns the number of times a `BaseTile` is executed
    //        along the direction of the shared memory row.
    template <typename Element, const int rows>
    DEVICE static constexpr int row_exec_count() {
        const int warps_per_row = tl::num_rows<WarpLayout>;
        const int base_tile_row = BaseTileShape<Element>::row;

        assert(rows % base_tile_row == 0 &&
               "The current implementation requires that the number of shared "
               "memory rows be divisible by the base tile row.\n");

        int count = 0;
        switch (kMode) {
            // Warps in the same columns (`warps_per_row` in total) repeatedly
            // load the shared memory rows. Therefore, `row_exec` is not divided
            // by warps_per_row.
            case WarpReuse::ColReuseCont:
            case WarpReuse::ColReuseCir:
                count = rows / base_tile_row;
                break;
            default:  // Cont, Cir, RowReuseCont, RowReuseCir hit this case.
                count = rows / base_tile_row / warps_per_row;
                break;
        }

        // Check to ensure that the count is not zero, which could be caused by
        // an incorrect combination of shared memory tile shape and warp layout.
        // TODO: This should actually be a static assert, but we're currently
        // using a runtime assert for implementation issues.
        assert(count);
        return count;
    }

    template <typename Element, const int cols>
    DEVICE static constexpr int col_exec_count() {
        const int warps_per_col = tl::num_cols<WarpLayout>;
        const int base_tile_col = BaseTileShape<Element>::col;

        assert(
            cols % base_tile_col == 0 &&
            "The number of shared memory columns must be divisible by the base "
            "tile column.\n");

        int count = 0;
        switch (kMode) {
            // Warps in the same rows (`warps_per_col` in total) repeatedly load
            // the shared memory columns. Therefore, `col_exec` is not divided
            // by `warps_per_col`.
            case WarpReuse::RowReuseCont:
            case WarpReuse::RowReuseCir:
                count = cols / base_tile_col;
                break;
            default:  // Cont, Cir, ColReuseCont, ColReuseCir hit this case.
                // FIXME(hotfix): 16 is a magic number for wmma tile
                count = cols / 16 / warps_per_col;
                break;
        }

        // Check to ensure that the count is not zero, which could be caused by
        // an incorrect combination of shared memory tile shape and warp layout.
        assert(count);
        return count;
    }

  private:
    // @brief the warp row that the current thread belongs to, based on the warp
    //        layout.
    DEVICE int warp_row_id() {
        /*
         * Example1: suppose the warp layout is RowMajor<2,2>, like this:
         * |-|-----|-----|
         * |0|warp0|warp1|
         * |-|-----|-----|
         * |1|warp2|warp3|
         * |-|-----|-----|, and the threadIdx is 67, then the warp row is 1.
         *
         * Example2: suppose the warp layout is ColMajor<2,2>, like this:
         * |-|-----|-----|
         * |0|warp0|warp2|
         * |-|-----|-----|
         * |1|warp1|warp3|
         * |-|-----|-----|, and the threadIdx is 67, then the warp row is 0.
         */
        int wid = threadIdx.x / warpSize;

        // FIXME(haruhi): The switch-case structure is currently inelegant.
        // Consider refining the code structure using more appropriate methods,
        // such as partial specialization.
        switch (tl::layout_type<WarpLayout>) {
            case tl::Layout::RowMajor:
                return wid / tl::num_cols<WarpLayout>;
            case tl::Layout::ColMajor:
                return wid % tl::num_rows<WarpLayout>;
            default:
                assert(false && "Not implemented yet.");
                return -1;
        }
    }

    // @brief: Returns the warp col that the current thread belongs to, based on
    //         the warp layout.
    DEVICE int warp_col_id() {
        /*
         * Example1: suppose the warp layout is RowMajor<2,2>, like this:
         * |-----|-----|
         * |  0  |  1  |
         * |-----|-----|
         * |warp0|warp1|
         * |-----|-----|
         * |warp2|warp3|
         * |-----|-----|, and the threadIdx is 67, then the warp col is 0.
         *
         * Example2: suppose the warp layout is ColMajor<2,2>, like this:
         * |-----|-----|
         * |  0  |  1  |
         * |-----|-----|
         * |warp0|warp2|
         * |-----|-----|
         * |warp1|warp3|
         * |-----|-----|, and the threadIdx is 67, then the warp row is 1.
         */
        int wid = threadIdx.x / warpSize;

        // FIXME(haruhi): The switch-case structure is currently inelegant.
        // Consider refining the code structure using more appropriate methods,
        // such as partial specialization.
        switch (tl::layout_type<WarpLayout>) {
            case tl::Layout::RowMajor:
                return wid % tl::num_cols<WarpLayout>;
            case tl::Layout::ColMajor:
                return wid / tl::num_rows<WarpLayout>;
            default:
                assert(false && "Not implemented yet.");
                return -1;
        }
    }
};

template <typename Shared, const tl::Layout type, CopyInst kCopyInst>
struct LoadTileImpl {
    using DType = typename Shared::DType;

    DEVICE void operator()(const DType* src, DType* dst, int row_exec,
                           int col_exec);
};

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

/// @brief partial specialization for row-major shared memory tile.
template <typename Shared>
struct LoadTileImpl<Shared, tl::Layout::RowMajor, CopyInst::LoadMat>
    : public LoadMatBase {
    using DType = typename Shared::DType;

    // strides to iterate over each 16x128-bits `BaseTile`.
    static constexpr int tile_rstride =
        BaseTileShape<DType>::row * Shared::kRowStride;
    static constexpr int tile_cstride = BaseTileShape<DType>::col;

    // row stride and col stride for a thread in a warp
    static constexpr int stride = traits::TraitsBase<DType>::kNumPerAccess;
    static constexpr int lane_rstride = Shared::kRowStride;
    static constexpr int lane_cstride = stride;

    DEVICE void operator()(const DType* src, DType* dst, int row_exec,
                           int col_exec) {
        int lane_row = lane_row_id();
        int lane_col = lane_col_id();

        const DType* data;
        for (int i = 0; i < row_exec; ++i) {
            for (int j = 0; j < col_exec; ++j) {
                // 2. advance pointer to the 16x128-bits `BaseTile` indexed by
                // (i, j).
                data = src + (i * tile_rstride + j * tile_cstride);

                // 3. advance the pointer to data accessed by the current thread
                // inside a 16x128-bits `BaseTile`.
                data += (lane_row * lane_rstride + lane_col * lane_cstride);

                // issue the hardware-backed memory access instruction.
                LoadMatBase::template ldmatrix<DType>(data, dst);

                // advance the pointer to store the next 16x128-bits `BaseTile`.
                dst += BaseTileShape<DType>::elem_per_thread;
            }
        }
    }
};

/// @brief partial specialization for column-major shared memory tile.
template <typename Shared>
struct LoadTileImpl<Shared, tl::Layout::ColMajor, CopyInst::LoadMat>
    : public LoadMatBase {
    using DType = typename Shared::DType;

    // strides to iterate over each 16x128-bits `BaseTile`.
    const int tile_rstride = BaseTileShape<DType>::row;
    const int tile_cstride = BaseTileShape<DType>::col * Shared::kColStride;

    // row stride and col stride for a thread in a warp
    static constexpr int stride = traits::TraitsBase<DType>::kNumPerAccess;
    const int lane_rstride = stride;
    const int lane_cstride = Shared::kColStride;

    DEVICE void operator()(const DType* src, DType* dst, int row_exec,
                           int col_exec) {
        // transpose the lane position if the shared memory is in column-major.
        // 16 threads are mapped to the strided dimension of the data while the
        // 2 threads are mapped to the contiguous dimension of the data.
        int lane_row = lane_col_id();
        int lane_col = lane_row_id();

        const DType* data;
        for (int i = 0; i < col_exec; ++i) {
            for (int j = 0; j < row_exec; ++j) {
                // 2. advance pointer to the 16x128-bits `BaseTile` indexed by
                // (i, j).
                data = src + (j * tile_rstride + i * tile_cstride);

                // 3. advance the pointer to data accessed by the current
                // thread inside a 16x128-bits `BaseTile`.
                data += (lane_row * lane_rstride + lane_col * lane_cstride);

                // issue the hardware-backed memory access instruction
                LoadMatBase::template ldmatrix<DType>(data, dst);

                // advance the pointer to store the next 16x128-bits `BaseTile`.
                dst += BaseTileShape<DType>::elem_per_thread;
            }
        }
    }
};

template <typename Shared, CopyInst kCopyInst>
struct StoreTileImpl {
    using DType = typename Shared::DType;

    DEVICE void operator()(const DType* src, DType* dst, int row_exec,
                           int col_exec);
};

template <typename Shared>
struct StoreTileImpl<Shared, CopyInst::LoadS32> {
    using DType = typename Shared::DType;

    using ThreadWmma = tile_layout::RowMajor<8, 4>;

    // FIXME(haruhi): Try to find a better way to organize these
    // hardware-dependent magic numbers.
    // The number 2 is interpreted as follows: 8x8 block is a fundamental
    // unit of a TCU instruction. Regardless of the output precision, a
    // single execution of WMMA stores 2 elements in each thread's local
    // register.
    static constexpr int stride = 2;
    static constexpr int rstride =
        Shared::type == tl::Layout::RowMajor
            ? tl::num_rows<ThreadWmma> * Shared::kRowStride
            : tl::num_rows<ThreadWmma>;
    static constexpr int cstride =
        Shared::type == tl::Layout::RowMajor
            ? stride * tl::num_cols<ThreadWmma>
            : stride * tl::num_cols<ThreadWmma> * Shared::kColStride;

    // strides to iterate over each 16x128-bits `Base Tile` in the shared
    // memory
    static constexpr int tile_rstride =  // row stride for a `Base Tile`
        Shared::type == tl::Layout::RowMajor
            ? BaseTileShape<DType>::row * Shared::kRowStride
            : BaseTileShape<DType>::row;
    static constexpr int tile_cstride =
        Shared::type == tl::Layout::RowMajor ? 16 : 16 * Shared::kColStride;

    // Given the lane position of the current thread, calculate the strides
    // needed to address the data within the 16x128-bit `BaseTile` for the
    // current thread.
    static constexpr int lane_rstride =
        Shared::type == tl::Layout::RowMajor ? Shared::kRowStride : stride;
    static constexpr int lane_cstride =
        Shared::type == tl::Layout::RowMajor ? stride : Shared::kColStride;

    DEVICE void operator()(const DType* src, DType* dst, int row_exec,
                           int col_exec) {
        DType* data;
        int lane_row = lane_row_id();
        int lane_col = lane_col_id();

        for (int i = 0; i < row_exec; ++i) {
            for (int j = 0; j < col_exec; ++j) {
                // 2. advance pointer to the 16x128-bits `BaseTile` indexed by
                // (i, j).
                data = dst + (i * tile_rstride + j * tile_cstride);
                // 3. advance the pointer to data accessed by the current thread
                // inside a 16x128-bits `Base Tile`.
                data += (lane_row * lane_rstride + lane_col * lane_cstride);

                // store the 16x128-bits `Base Tile` to shared memory
                store_base_tile(src, data, rstride, cstride);

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
    DEVICE void store_base_tile(const DType* src, DType* dst, int row_stride,
                                int col_stride) {
        int src_offset, dst_offset;
        // TODO(haruhi): comments the magic number
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                src_offset = i * 4 + j * 2;
                // Be cautious, j and i are swapped here. DONOT change this
                dst_offset = j * row_stride + i * col_stride;

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
          typename Base = detail::CopyBase<WarpLayout_, kMode_>>
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
        int row_exec = Base::template row_exec_count<DType, Shared::kRows>();
        int col_exec = Base::template col_exec_count<DType, Shared::kCols>();

        detail::LoadTileImpl<Shared, Shared::type, CopyInst::LoadMat> loader;
        loader(src_ptr, dst.mutable_data(), row_exec, col_exec);
    }
};

///@brief partial specialization for wmma 16x16x16, and LDSM32
template <typename Reg_, typename WarpLayout_,
          typename Base = detail::CopyBase<WarpLayout_, WarpReuse::Cont>>
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

        int row_exec = Base::template row_exec_count<DType, Shared::kRows>();
        int col_exec = Base::template col_exec_count<DType, Shared::kCols>();

        // 1. advance the pointer to input data to the current warp according to
        // warp reuse mode. During the store process, threads do not write to
        // the same shared memory location, thus the warp reuse mode is set to
        // `Cont`.
        int offset = Base::template get_warp_offset<Shared>();
        dst_ptr += offset;

        detail::StoreTileImpl<Shared, CopyInst::LoadS32> storer;
        storer(src.data(), dst_ptr, row_exec, col_exec);
    }
};

}  // namespace tiledcuda::cell::copy
