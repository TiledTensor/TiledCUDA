/**
 * @file copy_atom.hpp
 * @brief This file contains lightweight wrappers for hardware-accelerated copy
 *        instructions.
 */
#pragma once

#include "cell/traits/base.hpp"
#include "types/layout.hpp"

namespace tiledcuda::cell::copy::atom {
namespace tl = tile_layout;

template <typename Element>
requires std::is_same_v<Element, __half> ||
    std::is_same_v<Element, cutlass::half_t>
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
    //         |  | 0 |  1|
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
    DEVICE void ldmatrix(const DType* src, DType* dst) {
        uint32_t* reg = reinterpret_cast<uint32_t*>(dst);
        uint32_t smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(src));

        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
            : "r"(smem_addr));
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
        Shared::kType == tl::Layout::kRowMajor
            ? tl::num_rows<ThreadLayout> * Shared::kRowStride
            : tl::num_rows<ThreadLayout>;
    static constexpr int kCstride =
        Shared::kType == tl::Layout::kRowMajor
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
                src_offset = RegTile::kType == tl::Layout::kRowMajor
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

template <class Global, class Shared, const tl::Layout kType>
struct GlobalToSharedBaseTileLoader {
    using DType = Shared::DType;

    DEVICE void copy(const DType* src, DType* dst);
};

template <class Global, class Shared>
struct GlobalToSharedBaseTileLoader<Global, Shared, tl::Layout::kRowMajor> {
    using DType = Shared::DType;

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, DType>;
#endif

    // The macro kernel breaks down the entire copy operation into iterations
    // over 16x16 BaseTiles. To transfer a single BaseTile, threads in a warp
    // are arranged in a 16x2 row-major layout. Each thread uses 128-bit data in
    // a single access.
    static constexpr int kThreadsPerRow = 16;
    static constexpr int kThreadsPerCol = 2;

    static constexpr int kWarpSize = 32;

    static constexpr int kNumPerAccess =
        traits::TraitsBase<DType>::kNumPerAccess;

    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kExecCount =
        BaseShape::kCols / kNumPerAccess / kThreadsPerCol;

    static_assert(
        kExecCount == 1,
        "The current implementation requires that number of elements per "
        "access should be equal to the number of columns in the BaseTile.");

    using BaseTileGlobalLayout =
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<Int<Global::kRowStride>, _1>>;

    using BaseTileSharedLayout = tl::SharedLayoutWrapper<Shared>::Layout;

    using TiledCopy = decltype(make_tiled_copy(
        CopyInst{},
        cute::Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
                     Stride<Int<kThreadsPerCol>, _1>>{},
        cute::Layout<Shape<_1, Int<kNumPerAccess>>>{}));

    using DataLayoutPerThread = cute::Layout<Shape<Int<kNumPerAccess>, _1>,
                                             Stride<_1, Int<kNumPerAccess>>>;

    DEVICE GlobalToSharedBaseTileLoader() : tiled_copy_(TiledCopy{}) {}

    DEVICE void copy(const DType* src, DType* dst) {
        auto src_tensor = make_tensor(make_gmem_ptr(src), data_layout_);
        auto dst_tensor = make_tensor(make_smem_ptr(dst), data_layout_);

        cute::copy(tiled_copy_, src_tensor, dst_tensor);
    }

    /// @brief returns the lane row of the current thread within a warp.
    DEVICE int lane_row_id() {
        int lane_id = threadIdx.x % kWarpSize;
        return lane_id / kThreadsPerCol;
    }

    /// @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_col_id() {
        int lane_id = threadIdx.x % kWarpSize;
        return lane_id % kThreadsPerCol;
    }

  private:
    DataLayoutPerThread data_layout_;
    TiledCopy tiled_copy_;
};

}  // namespace tiledcuda::cell::copy::atom
