/**
 * @file copy_atom.hpp
 * @brief This file contains thin wrappers for hardware-baked copy instruction.
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
        return lane_id % tl::num_rows<ThreadLayout>;
    }

    // @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_col_id() {
        int lane_id = threadIdx.x % warpSize;
        return lane_id / tl::num_rows<ThreadLayout>;
    }

    /// @brief a thin wrapper for executing a single ldmatrix instruction.
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
    using ThreadLayout = tile_layout::RowMajor<8, 4>;

    // FIXME(haruhi): Try to find a better way to organize these
    // hardware-dependent magic numbers.
    // The number 2 is interpreted as follows: 8x8 block is a fundamental
    // unit of a TCU instruction. Regardless of the output precision, a single
    // execution of WMMA stores 2 elements in each thread's local register.
    static constexpr int kStride = 2;
    static constexpr int kRstride =
        Shared::type == tl::Layout::RowMajor
            ? tl::num_rows<ThreadLayout> * Shared::kRowStride
            : tl::num_rows<ThreadLayout>;
    static constexpr int kCstride =
        Shared::type == tl::Layout::RowMajor
            ? kStride * tl::num_cols<ThreadLayout>
            : kStride * tl::num_cols<ThreadLayout> * Shared::kColStride;

    DEVICE int lane_row_id() {
        return (threadIdx.x % warpSize) / tl::num_cols<ThreadLayout>;
    }

    DEVICE int lane_col_id() {
        return (threadIdx.x % warpSize) % tl::num_cols<ThreadLayout>;
    }

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
};

}  // namespace tiledcuda::cell::copy::atom
