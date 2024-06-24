#pragma once

#include "cell/copy/constants.hpp"
#include "types/mod.hpp"
#include "util/print.hpp"

namespace tiledcuda::cell::copy {

namespace {  // helper functions
template <typename Element>
DEVICE void ldmatrix(const Element* src, Element* dst) {
    uint32_t* reg = reinterpret_cast<uint32_t*>(dst);
    uint32_t smem_addr = __cvta_generic_to_shared(src);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[2]), "=r"(reg[1]), "=r"(reg[3])
        : "r"(smem_addr));
}

/// @brief the warp row that the current thread belongs to, based on the warp
///        layout.
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

/// @brief: Returns the warp col that the current thread belongs to, based on
///         the warp layout.
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

/// @brief This function returns the lane row of the current thread within a
///        warp. We assume that the threads in a warp are arranged as follows (a
///        16 x 2 column-major):
/*         |  | 0 |  1|
 *         |--|---|---|
 *         |0 | 0 | 16|
 *         |1 | 2 | 17|
 *         |2 | 4 | 18|
 *         |  |...|...|
 *         |15| 15| 31|
 *  Example: if threadIdx.x is 43, then its lane row is 8, lane col is 0.
 */
DEVICE int lane_row_id() {
    int lane_id = threadIdx.x % warpSize;  // thread index inside a warp
    return lane_id % 16;
}

/// @brief This function returns the lane col of the current thread within a
///        warp. We assume that the threads in a warp are arranged as explained
///        above.
DEVICE int lane_col_id() {
    int lane_id = threadIdx.x % warpSize;  // thread index inside a warp
    return lane_id / 16;
}

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

        // strides to iterate over each 16x128-bits base tile.
        const int tile_rstride =  // row stride for a base tile
            row_major ? BaseTileShape<DType>::row * Shared::kRowStride
                      : BaseTileShape<DType>::row;
        const int tile_cstride =  // col stride for a base tile
            row_major ? BaseTileShape<DType>::col
                      : BaseTileShape<DType>::col * Shared::kColStride;

        // row stride and col stride for a thread in a warp
        const int lane_rstride = row_major ? Shared::kRowStride : 8;
        const int lane_cstride = row_major ? 8 : Shared::kColStride;

        int lane_row = lane_row_id();
        int lane_col = lane_col_id();

        const DType* data;
        for (int i = 0; i < row_exec; ++i) {
            for (int j = 0; j < col_exec; ++j) {
                // 2. advance pointer to the 16x128-bits `BaseTile` indexed by
                // (i, j).
                data = src_ptr + (i * tile_rstride + j * tile_cstride);
                // 3. advance the pointer to data accessed by the current thread
                // inside a 16x16 base tile.
                data += (lane_row * lane_rstride + lane_col * lane_cstride);

                // issue the hardware-backed memory access instruction
                ldmatrix(data, dst_ptr);

                // advance the pointer to store the next 16x16 base tile.
                dst_ptr += BaseTileShape<DType>::elem_per_thread;
            }
        }
        dst.dump_value();
    }
};

// functor to copy data from shared memory to register file.
template <typename Reg, typename Shared, typename InstShape,
          RegLayout kRegLayout, CopyInst kCopyInst>
struct RegToShared {
    DEVICE void operator()(const Reg& src, Shared& dst);
};

// partial specialization for wmma 16x16x16, and LDSM32
template <typename Reg, typename Shared>
struct RegToShared<Reg, Shared, InstShape<16, 16, 16>, RegLayout::TileWMMA,
                   CopyInst::LoadS32> {
    DEVICE void operator()(const Reg& src, Shared& dst) {}
};

template <typename Reg, typename Shared>
DEVICE void copy_tile_r2s(const Reg& src, Shared& dst) {
    using Copy = RegToShared<Reg, Shared, InstShape<16, 16, 16>,
                             RegLayout::TileWMMA, CopyInst::LoadS32>;
    Copy copy;
    copy(src, dst);
}

}  // namespace tiledcuda::cell::copy
