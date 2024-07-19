/**
 * @file copy_atom.hpp
 * @brief This file contains lightweight wrappers for hardware-accelerated copy
 *        instructions.
 */
#pragma once

#include "types/layout.hpp"

namespace tiledcuda::cell::copy::atom {

namespace tl = tiledcuda::cell::tile_layout;

template <typename Element>
struct LoadMatBase {
    using DType = Element;

    using ThreadLayout = tile_layout::ColMajor<16, 2>;

    static constexpr int kAccessInBits = 128;  // 128 bits
    static constexpr int kElmentBits = sizeof(DType) * 8;
    static constexpr int kNumPerAccess = kAccessInBits / kElmentBits;

    /// @brief returns the lane row of the current thread within a warp.
    //         For an example, in ldmatrix, threads in a warp are arranged as
    //         follows (a 16 x 2 column-major):
    //
    // .       |  | 0 |  1|
    //         |--|---|---|
    //         |0 | 0 | 16|
    //         |1 | 2 | 17|
    //         |2 | 4 | 18|
    //         |  |...|...|
    //         |15| 15| 31|
    //
    //         if threadIdx.x is 43, then its lane row is 8, lane col is 0.
    DEVICE int lane_row_id() {
        int lane_id = threadIdx.x % warpSize;
        return lane_id % tl::num_rows<ThreadLayout>;
    }

    /// @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_col_id() {
        int lane_id = threadIdx.x % warpSize;
        return lane_id / tl::num_rows<ThreadLayout>;
    }

    /// @brief a thin wrapper for executing ldmatrix instruction to load a
    ///        `16x16` tile to register.
    template <const int kRows, const int kCols>
    DEVICE void ldmatrix(const DType* src, DType* dst) {
        uint32_t* reg = reinterpret_cast<uint32_t*>(dst);
        uint32_t smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(src));

        for (int i = 0; i < kRows; ++i) {
            for (int j = 0; j < kCols; ++j) {
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
                    : "r"(smem_addr));

                reg += 4;
                smem_addr += 1;
            }
        }
    }
};

template <typename Shared>
struct StoreMmmaBase {
    using DType = Shared::DType;

    // the thread layout for wmma's output tile.
    using ThreadLayout = tile_layout::RowMajor<8, 4>;

    // in the output of a wmma tile, each thread stores four segments in 2x2
    // layout, and each fragment contains 2 elements
    static constexpr int kElemPerSeg = 2;
    static constexpr int kSegRows = 2;
    static constexpr int kSegCols = 2;
    static constexpr int kElemPerRow = kElemPerSeg * kSegCols;

    // Each thread stores 8 numbers in a 16x16 `BaseTile`. These 8 numbers are
    // split into 4 segments, with each segment containing 2 numbers. `kRStride`
    // and `kCStride` calculate the row and column strides, respectively, when
    // iterating over these 4 segments in shared memory.
    static constexpr int kRstride =
        Shared::type == tl::Layout::RowMajor
            ? tl::num_rows<ThreadLayout> * Shared::kRowStride
            : tl::num_rows<ThreadLayout>;
    static constexpr int kCstride =
        Shared::type == tl::Layout::RowMajor
            ? kElemPerSeg * tl::num_cols<ThreadLayout>
            : kElemPerSeg * tl::num_cols<ThreadLayout> * Shared::kColStride;

    DEVICE int lane_row_id() {
        return (threadIdx.x % warpSize) / tl::num_cols<ThreadLayout>;
    }

    DEVICE int lane_col_id() {
        return (threadIdx.x % warpSize) % tl::num_cols<ThreadLayout>;
    }

    /// @brief: Store the `16x16` tensor core WMMA output register tile and
    ///         convert it into a comprehensive shared memory layout (row-major
    ///         or column-major).
    template <typename RegTile>
    DEVICE void store_base_tile(const RegTile& src, DType* dst) {
        const DType* data = src.data();

        int src_offset, dst_offset;
        for (int i = 0; i < kSegRows; ++i) {
            for (int j = 0; j < kSegCols; ++j) {
                src_offset = RegTile::kType == tl::Layout::RowMajor
                                 ? i * kElemPerRow + j * kElemPerSeg
                                 : j * kElemPerRow + i * kElemPerSeg;

                // Be cautious, j and i are swapped here. DONOT change this
                dst_offset = j * kRstride + i * kCstride;

                dst[dst_offset] = data[src_offset];
                dst[dst_offset + 1] = data[src_offset + 1];
            }
        }
    }
};

}  // namespace tiledcuda::cell::copy::atom
