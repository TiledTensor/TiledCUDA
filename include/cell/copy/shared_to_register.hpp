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
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(smem_addr));
}

/// @brief the warp row that the current thread belongs to, based on the warp
/// layout.
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
/// the warp layout.
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

template <typename Reg_, typename WarpLayout_>
struct SharedToRegLoader<Reg_, WarpLayout_, WarpReuse::RowReuseCont,
                         CopyInst::LoadMat> {
    using Reg = Reg_;
    using WarpLayout = WarpLayout_;
    static constexpr WarpReuse kMode = WarpReuse::RowReuseCont;

    using DType = typename Reg::DType;

    template <typename Shared>
    DEVICE void operator()(const Shared& src, Reg& dst) {
        static_assert(std::is_same<typename Shared::DType, DType>::value,
                      "The data type of Shared and Reg must be the same.");
        // print_tile(src.data(), src.layout());

        static_assert(Shared::kRows % tl::num_rows<WarpLayout> == 0,
                      "The current implementation requires Shared::kRows must "
                      "be divisible by tl::num_rows<WarpLayout>");
        static_assert(Shared::kCols % tl::num_cols<WarpLayout> == 0,
                      "The current implementation requires Shared::kCols must "
                      "be divisible by tl::num_cols<WarpLayout>");

        // how many times `BaseTile` are executed alongs rows and cols by the
        // current thread
        const int row_exec = row_exec_count<DType, Shared::kRows,
                                            tl::num_rows<WarpLayout>, kMode>();
        const int col_exec = col_exec_count<DType, Shared::kCols,
                                            tl::num_cols<WarpLayout>, kMode>();

        // Tile shape for a single warp
        using WarpTileShape = TileShape<row_exec * BaseTileShape<DType>::row,
                                        col_exec * BaseTileShape<DType>::col>;

        int warp_row = warp_row_id<WarpLayout>();
        int warp_col = warp_col_id<WarpLayout>();

        if (thread(57)) {
            // printf("\nwarp shape :");
            // print(WarpTileShape{});

            printf("\n\nrow_exec: %d, col_exec: %d\n", row_exec, col_exec);
            printf("warp_row: %d, warp_col: %d\n", warp_row, warp_col);
        }

        const DType* src_ptr = src.data();
        DType* dst_ptr = dst.mutable_data();

        for (int i = 0; i < row_exec; ++i) {
            for (int j = 0; j < col_exec; ++j) {
                ldmatrix(src_ptr, dst_ptr);

                dst_ptr += BaseTileShape<DType>::elem_per_thread;
                break;
            }
            break;
        }
    }
};

template <typename Reg_, typename WarpLayout_>
struct SharedToRegLoader<Reg_, WarpLayout_, WarpReuse::ColReuseCont,
                         CopyInst::LoadMat> {
    using Reg = Reg_;
    using WarpLayout = WarpLayout_;
    static constexpr WarpReuse kMode = WarpReuse::ColReuseCont;

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

        int row_exec = row_exec_count<DType, Shared::kRows,
                                      tl::num_rows<WarpLayout>, kMode>();
        int col_exec = col_exec_count<DType, Shared::kCols,
                                      tl::num_cols<WarpLayout>, kMode>();

        int warp_row = warp_row_id<WarpLayout>();
        int warp_col = warp_col_id<WarpLayout>();

        if (thread0()) {
            printf("\n\trow_exec: %d, col_exec: %d\n", row_exec, col_exec);
        }
    }
};

namespace detail {
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
}  // namespace detail

template <typename Reg, typename Shared>
DEVICE void copy_tile_r2s(const Reg& src, Shared& dst) {
    using Copy = detail::RegToShared<Reg, Shared, InstShape<16, 16, 16>,
                                     RegLayout::TileWMMA, CopyInst::LoadS32>;
    Copy copy;
    copy(src, dst);
}

}  // namespace tiledcuda::cell::copy
