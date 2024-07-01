#pragma once

#include "cell/copy/constants.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

using namespace tiledcuda::cell::traits;

namespace {  // helper functions
template <typename Element>
DEVICE void ldmatrix(const Element* src, Element* dst) {
    uint32_t* reg = reinterpret_cast<uint32_t*>(dst);
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(smem_addr));
}

// @brief the warp row that the current thread belongs to, based on the warp
//        layout.
template <typename WarpLayout>
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
    int warp_row = tl::is_rowmajor<WarpLayout> ? wid / tl::num_cols<WarpLayout>
                                               : wid % tl::num_rows<WarpLayout>;
    return warp_row;
}

// @brief: Returns the warp col that the current thread belongs to, based on
//         the warp layout.
template <typename WarpLayout>
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
    int warp_col = tl::is_rowmajor<WarpLayout> ? wid % tl::num_cols<WarpLayout>
                                               : wid / tl::num_rows<WarpLayout>;
    return warp_col;
}

// @brief This function returns the lane row of the current thread within a
//        warp. NOTE: this function is only valid for the thread layout in
//        `ldmatrix` or `stmatrix`. We assume that the threads in a warp are
//        arranged as follows (a 16 x 2 column-major):
//        |  | 0 |  1|
//        |--|---|---|
//        |0 | 0 | 16|
//        |1 | 2 | 17|
//        |2 | 4 | 18|
//        |  |...|...|
//        |15| 15| 31|
// Example: if threadIdx.x is 43, then its lane row is 8, lane col is 0.
template <typename Layout>
DEVICE int lane_row_id() {
    int lane_id = threadIdx.x % warpSize;  // thread index inside a warp
    return tl::is_rowmajor<Layout> ? lane_id / tl::num_cols<Layout>
                                   : lane_id % tl::num_rows<Layout>;
}

// @brief This function returns the lane col of the current thread within a
//        warp. NOTE: this function is only valid for the thread layout in
//        `ldmatrix` or `stmatrix`. We assume that the threads in a warp are
//        arranged as explained above.
template <typename Layout>
DEVICE int lane_col_id() {
    int lane_id = threadIdx.x % warpSize;  // thread index inside a warp
    return tl::is_rowmajor<Layout> ? lane_id % tl::num_cols<Layout>
                                   : lane_id / tl::num_rows<Layout>;
}

// @brief This function returns the offset to the start position of the current
//        warp in the shared memory according to the warp reuse mode.
template <const WarpReuse kMode, typename Shared, typename WarpLayout>
DEVICE int get_warp_offset() {
    constexpr static bool row_major = Shared::kIsRowMajor;

    // Tile shape for a single warp
    constexpr static int warp_shape_row =
        Shared::kRows / tl::num_rows<WarpLayout>;
    constexpr static int warp_shape_col =
        Shared::kCols / tl::num_cols<WarpLayout>;

    constexpr static int warp_rstride =
        row_major ? Shared::kRowStride * warp_shape_row : warp_shape_row;
    constexpr static int warp_cstride =
        row_major ? warp_shape_col : Shared::kColStride * warp_shape_col;

    int offset = 0;
    switch (kMode) {
        case WarpReuse::Cont: {
            // In 'Cont' mode, all warps evenly divide the large data tiles
            // and load the data without repeated loads.
            int warp_row = warp_row_id<WarpLayout>();
            int warp_col = warp_col_id<WarpLayout>();
            offset = warp_row * warp_rstride + warp_col * warp_cstride;
            break;
        }
        case WarpReuse::RowReuseCont: {
            // In the `RowReuseCont` mode, warps in a same row repeatedly load
            // the same data, and each warp loads a continuous chunks of data.
            int warp_row = warp_row_id<WarpLayout>();
            offset = warp_row * warp_rstride;
            break;
        }
        case WarpReuse::ColReuseCont: {
            // In the `ColReuseCont` mode, warps in a same column repeatedly
            // load the same data, and each warp loads a continuous chunks of
            // data.
            int warp_col = warp_col_id<WarpLayout>();
            offset = warp_col * warp_cstride;
            break;
        }
        default: {
            // simulate "the not implemented error"
            assert(false && "Not implemented yet.");
            break;
        }
    }
    return offset;
}

// @brief This function returns the number of times a `BaseTile` is executed
//        along the direction of the shared memory row.
template <typename Element, const int rows, const int warps_per_row,
          const WarpReuse mode>
DEVICE static constexpr int row_exec_count() {
    const int base_tile_row = BaseTileShape<Element>::row;

    assert(rows % base_tile_row == 0 &&
           "The current implementation requires that the number of shared "
           "memory rows be divisible by the base tile row.\n");

    int count = 0;
    switch (mode) {
        /// Warps in the same columns (`warps_per_row` in total) repeatedly load
        /// the shared memory rows. Therefore, `row_exec` is not divided by
        /// warps_per_row.
        case WarpReuse::ColReuseCont:
        case WarpReuse::ColReuseCir:
            count = rows / base_tile_row;
            break;
        default:  // Cont, Cir, RowReuseCont, RowReuseCir hit this case.
            count = rows / base_tile_row / warps_per_row;
            break;
    }

    // Check to ensure that the count is not zero, which could be caused by an
    // incorrect combination of shared memory tile shape and warp layout.
    // TODO: This should actually be a static assert, but we're
    // currently using a runtime assert for implementation issues.
    assert(count);
    return count;
}

template <typename Element, const int cols, const int warps_per_col,
          const WarpReuse mode>
DEVICE static constexpr int col_exec_count() {
    const int base_tile_col = BaseTileShape<Element>::col;

    assert(cols % base_tile_col == 0 &&
           "The number of shared memory columns must be divisible by the base "
           "tile column.\n");

    int count = 0;
    switch (mode) {
        /// Warps in the same rows (`warps_per_col` in total) repeatedly load
        /// the shared memory columns. Therefore, `col_exec` is not divided by
        /// `warps_per_col`.
        case WarpReuse::RowReuseCont:
        case WarpReuse::RowReuseCir:
            count = cols / base_tile_col;
            break;
        default:  // Cont, Cir, ColReuseCont, ColReuseCir hit this case.
            count = cols / base_tile_col / warps_per_col;
            break;
    }

    // Check to ensure that the count is not zero, which could be caused by an
    // incorrect combination of shared memory tile shape and warp layout.
    assert(count);
    return count;
}

}  // namespace

/// @brief The functor that copys data from shared memory to register file.
///        Shared memory tile is not declared as template parameter, because we
///        assume it is computed from runtime value by `TileIterator`.
template <typename Reg,
          typename WarpLayout,  //
          WarpReuse kMode,      //
          CopyInst kCopyInst>
struct SharedToRegLoader {
    template <typename Shared>
    DEVICE void operator()(const Shared& src, Reg& dst);
};

/// @brief partial specialization for loading data from shared memory to
///        register file using `ldmatrix`.
template <typename Reg_, typename WarpLayout_, const WarpReuse kMode>
struct SharedToRegLoader<Reg_, WarpLayout_, kMode, CopyInst::LoadMat> {
    using Reg = Reg_;
    using WarpLayout = WarpLayout_;
    using DType = typename Reg::DType;

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

        const bool row_major = Shared::kIsRowMajor;
        const DType* src_ptr = src.data();    // pointer for shared memory tile
        DType* dst_ptr = dst.mutable_data();  // pointer for register tile

        // how many times a `BaseTile` is executed along the row and column
        // direction.
        int row_exec = row_exec_count<DType, Shared::kRows,
                                      tl::num_rows<WarpLayout>, kMode>();
        int col_exec = col_exec_count<DType, Shared::kCols,
                                      tl::num_cols<WarpLayout>, kMode>();

        // 1. advance the pointer to input data to the current warp according to
        // warp reuse mode.
        src_ptr += get_warp_offset<kMode, Shared, WarpLayout>();

        // strides to iterate over each 16x128-bits `Base Tile`.
        const int tile_rstride =  // row stride for a `Base Tile`
            row_major ? BaseTileShape<DType>::row * Shared::kRowStride
                      : BaseTileShape<DType>::row;
        const int tile_cstride =  // col stride for a `Base Tile`
            row_major ? BaseTileShape<DType>::col
                      : BaseTileShape<DType>::col * Shared::kColStride;

        // row stride and col stride for a thread in a warp
        static constexpr int stride = traits::TraitsBase<DType>::kNumPerAccess;
        const int lane_rstride = row_major ? Shared::kRowStride : stride;
        const int lane_cstride = row_major ? stride : Shared::kColStride;

        int lane_row = lane_row_id<traits::ThreadLdmatrix>();
        int lane_col = lane_col_id<traits::ThreadLdmatrix>();

        if (!row_major) {
            // transpose the lane position if the shared memory is in
            // column-major. 16 threads are mapped to the strided dimension of
            // the data while the 2 threads are mapped to the contiguous
            // dimension of the data.
            int tmp = lane_row;
            lane_row = lane_col;
            lane_col = tmp;
        }

        const DType* data;
        for (int i = 0; i < row_exec; ++i) {
            for (int j = 0; j < col_exec; ++j) {
                // 2. advance pointer to the 16x128-bits `BaseTile` indexed by
                // (i, j).
                data = src_ptr + (i * tile_rstride + j * tile_cstride);

                // 3. advance the pointer to data accessed by the current thread
                // inside a 16x128-bits `Base Tile`.
                data += (lane_row * lane_rstride + lane_col * lane_cstride);

                // issue the hardware-backed memory access instruction
                ldmatrix(data, dst_ptr);

                // advance the pointer to store the next 16x128-bits
                // `Base Tile`.
                dst_ptr += BaseTileShape<DType>::elem_per_thread;
            }
        }
    }
};

///@brief A functor that implements threads in a CTA to cooperatively store a
///       register tile in shared memory.
template <typename Reg, typename WarpLayout, RegLayout kRegLayout,
          CopyInst kCopyInst>
struct RegToSharedStorer {
    // shared memory tile may be computed from runtime value, thus leave it as a
    // parameter instead of template parameter.
    template <typename Shared>
    DEVICE void operator()(const Reg& src, Shared& dst);
};

///@brief partial specialization for wmma 16x16x16, and LDSM32
template <typename Reg_, typename WarpLayout_>
struct RegToSharedStorer<Reg_, WarpLayout_, RegLayout::WMMA_m16n16k16,
                         CopyInst::LoadS32> {
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

        const bool row_major = Shared::kIsRowMajor;  // get the layout of shared
        const DType* src_ptr = src.data();    // pointer for register tile
        DType* dst_ptr = dst.mutable_data();  // pointer for shared memory tile

        int row_exec =
            row_exec_count<DType, Shared::kRows, tl::num_rows<WarpLayout>,
                           WarpReuse::Cont>();
        int col_exec =
            row_exec_count<DType, Shared::kCols, tl::num_cols<WarpLayout>,
                           WarpReuse::Cont>();

        // 1. advance the pointer to input data to the current warp according to
        // warp reuse mode. During the store process, threads do not write to
        // the same shared memory location, thus the warp reuse mode is set to
        // `Cont`.
        int offset = get_warp_offset<WarpReuse::Cont, Shared, WarpLayout>();
        dst_ptr += offset;

        // FIXME(haruhi): Try to find a better way to organize these
        // hardware-dependent magic numbers.
        // The number 2 is interpreted as follows: 8x8 block is a fundamental
        // unit of a TCU instruction. Regardless of the output precision, a
        // single execution of WMMA stores 2 elements in each thread's local
        // register.
        static constexpr int stride = 2;
        static constexpr int rstride =
            row_major ? tl::num_rows<ThreadWmma> * Shared::kRowStride
                      : tl::num_rows<ThreadWmma>;
        static constexpr int cstride =
            row_major ? stride * tl::num_cols<ThreadWmma>
                      : stride * tl::num_cols<ThreadWmma> * Shared::kColStride;

        // TODO: Improve this is a naive implementation that does not use
        // vectorized access in future. To use vectorized access, we need to
        // know the return type of WMMA.
        auto store_base_tile = [&](const DType* src, DType* dst) {
            int src_offset, dst_offset;
            // TODO(haruhi): comments the magic number
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    src_offset = i * 4 + j * 2;
                    // Be cautious, j and i are swapped here. DONOT change this
                    dst_offset = j * rstride + i * cstride;

                    dst[dst_offset] = src[src_offset];
                    dst[dst_offset + 1] = src[src_offset + 1];
                }
            }
        };

        // strides to iterate over each 16x128-bits `Base Tile` in the shared
        // memory
        const int tile_rstride =  // row stride for a `Base Tile`
            row_major ? BaseTileShape<DType>::row * Shared::kRowStride
                      : BaseTileShape<DType>::row;
        const int tile_cstride = row_major ? 16 : 16 * Shared::kColStride;

        // Given the lane position of the current thread, calculate the strides
        // needed to address the data within the 16x128-bit `Base Tile` for the
        // current thread.
        const int lane_rstride = row_major ? Shared::kRowStride : stride;
        const int lane_cstride = row_major ? stride : Shared::kColStride;

        DType* data;
        int lane_row = lane_row_id<traits::ThreadWmma>();
        int lane_col = lane_col_id<traits::ThreadWmma>();

        for (int i = 0; i < row_exec; ++i) {
            for (int j = 0; j < col_exec; ++j) {
                // 2. advance pointer to the 16x128-bits `BaseTile` indexed by
                // (i, j).
                data = dst_ptr + (i * tile_rstride + j * tile_cstride);
                // 3. advance the pointer to data accessed by the current thread
                // inside a 16x128-bits `Base Tile`.
                data += (lane_row * lane_rstride + lane_col * lane_cstride);

                // store the 16x128-bits `Base Tile` to shared memory
                store_base_tile(src_ptr, data);

                src_ptr += 8;  // FIXME(haruhi): the magic number is dangerous.
            }
        }
    }
};

}  // namespace tiledcuda::cell::copy
