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

template <typename DType_, const int kSegRows_, const int kSegCols_,
          const int kRstride_, const int kCstride_, const size_t kBitPerAccess>
struct BaseTileStorer;

template <typename DType_, const int kSegRows_, const int kSegCols_,
          const int kRstride_, const int kCstride_>
struct BaseTileStorer<DType_, kSegRows_, kSegCols_, kRstride_, kCstride_, 32> {
    using DType = DType_;
    static constexpr int kSegRows = kSegRows_;
    static constexpr int kSegCols = kSegCols_;
    static constexpr int kRstride = kRstride_;
    static constexpr int kCstride = kCstride_;

    DEVICE void operator()(const DType* src_, DType* dst_) {
        const int* src = reinterpret_cast<const int*>(src_);
        int* dst = reinterpret_cast<int*>(dst_);

#pragma unroll
        for (int i = 0; i < kSegCols; ++i) {
#pragma unroll
            for (int j = 0; j < kSegRows; ++j) {
                dst[i * kRstride + j * kCstride] = src[j * kSegCols + i];
            }
        }
    }
};

template <typename DType_, const int kSegRows_, const int kSegCols_,
          const int kRstride_, const int kCstride_, const size_t kBitPerAccess>
struct BaseTileStorer;

template <typename DType_, const int kSegRows_, const int kSegCols_,
          const int kRstride_, const int kCstride_>
struct BaseTileStorer<DType_, kSegRows_, kSegCols_, kRstride_, kCstride_, 64> {
    using DType = DType_;
    static constexpr int kSegRows = kSegRows_;
    static constexpr int kSegCols = kSegCols_;
    static constexpr int kRstride = kRstride_;
    static constexpr int kCstride = kCstride_;

    DEVICE void operator()(const DType* src_, DType* dst_) {
        const int2* src = reinterpret_cast<const int2*>(src_);
        int2* dst = reinterpret_cast<int2*>(dst_);

#pragma unroll
        for (int i = 0; i < kSegCols; ++i) {
#pragma unroll
            for (int j = 0; j < kSegRows; ++j) {
                dst[i * kRstride + j * kCstride] = src[j * kSegCols + i];
            }
        }
    }
};

template <class Global, class Shared, const tl::Layout kType>
struct GlobalToSharedBaseTileLoader {
    using DType = Shared::DType;

    DEVICE void copy(const DType* src, DType* dst);
};

/// @brief  Implement loading a `16x16` BaseTile from global memory to shared
///         memory.
template <class Global, class Shared>
struct GlobalToSharedBaseTileLoader<Global, Shared, tl::Layout::kRowMajor> {
    using DType = Shared::DType;

    // NOTE: Please keep this thread layout striclty consistent with the thread
    // layout for ldmatrix.
    // The macro kernel breaks down the entire copy operation into iterations
    // over 16x16 BaseTiles. To transfer a single BaseTile, threads in a warp
    // are arranged in a 16x2 row-major layout. Each thread uses 128-bit data in
    // a single access.
    using ThreadLayout = tile_layout::ColMajor<16, 2>;
    static constexpr int kThreadsPerRow = tl::num_rows<ThreadLayout>;
    static constexpr int kThreadsPerCol = tl::num_cols<ThreadLayout>;

    static constexpr int kWarpSize = 32;

    static constexpr int kNumPerAccess =
        traits::TraitsBase<DType>::kNumPerAccess;

    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kExecCount =
        BaseShape::kCols / (kNumPerAccess * kThreadsPerCol);

    static_assert(
        kExecCount == 1,
        "The current implementation requires that number of elements per "
        "access should be equal to the number of columns in the BaseTile.");

    using BaseTileGlobalLayout =
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<Int<Global::kRowStride>, _1>>;

    using BaseTileSharedLayout = tl::SharedLayoutWrapper<Shared>::Layout;

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, DType>;
#endif
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

    DEVICE int lane_row_id() {
        int lane_id = threadIdx.x % kWarpSize;
        return lane_id % tl::num_rows<ThreadLayout>;
    }

    /// @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_col_id() {
        int lane_id = threadIdx.x % kWarpSize;
        return lane_id / tl::num_rows<ThreadLayout>;
    }

  private:
    DataLayoutPerThread data_layout_;
    TiledCopy tiled_copy_;
};

/// @brief  Implement loading a `16x16` BaseTile from global memory to shared
///         memory.
template <class Global, class Shared>
struct GlobalToSharedBaseTileLoader<Global, Shared, tl::Layout::kColMajor> {
    using DType = Shared::DType;

    // The macro kernel breaks down the entire copy operation into iterations
    // over 16x16 BaseTiles. To transfer a single BaseTile, threads in a warp
    // are arranged in a 16x2 row-major layout. Each thread uses 128-bit data in
    // a single access.
    using ThreadLayout = tile_layout::RowMajor<2, 16>;
    static constexpr int kThreadsPerRow = tl::num_rows<ThreadLayout>;
    static constexpr int kThreadsPerCol = tl::num_cols<ThreadLayout>;

    static constexpr int kWarpSize = 32;

    static constexpr int kNumPerAccess =
        traits::TraitsBase<DType>::kNumPerAccess;

    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kExecCount =
        BaseShape::kRows / (kNumPerAccess * kThreadsPerRow);

    static_assert(
        kExecCount == 1,
        "The current implementation requires that number of elements per "
        "access should be equal to the number of columns in the BaseTile.");

    // create CuTe's compatible column-major layout for the global memory.
    using BaseTileGlobalLayout =
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<_1, Int<Global::kColStride>>>;

    using BaseTileSharedLayout = tl::SharedLayoutWrapper<Shared>::Layout;

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, DType>;
#endif
    using TiledCopy = decltype(make_tiled_copy(
        CopyInst{},
        cute::Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
                     Stride<Int<kThreadsPerCol>, _1>>{},
        cute::Layout<Shape<Int<kNumPerAccess>, _1>>{}));

    using DataLayoutPerThread = cute::Layout<Shape<Int<kNumPerAccess>, _1>,
                                             Stride<_1, Int<kNumPerAccess>>>;

    DEVICE GlobalToSharedBaseTileLoader() : tiled_copy_(TiledCopy{}) {}

    DEVICE void copy(const DType* src, DType* dst) {
        auto src_tensor = make_tensor(make_gmem_ptr(src), data_layout_);
        auto dst_tensor = make_tensor(make_smem_ptr(dst), data_layout_);

        cute::copy(tiled_copy_, src_tensor, dst_tensor);
    }

    DEVICE int lane_col_id() {
        int lane_id = threadIdx.x % kWarpSize;
        return lane_id % tl::num_cols<ThreadLayout>;
    }

    /// @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_row_id() {
        int lane_id = threadIdx.x % kWarpSize;
        return lane_id / tl::num_cols<ThreadLayout>;
    }

  private:
    DataLayoutPerThread data_layout_;
    TiledCopy tiled_copy_;
};

template <class Shared, class Global, const tl::Layout kType>
struct SharedToGlobalBaseTileStorer {
    using DType = Shared::DType;

    DEVICE void copy(const DType* src, DType* dst);
};

template <class Shared, class Global>
struct SharedToGlobalBaseTileStorer<Shared, Global, tl::Layout::kRowMajor> {
    using DType = Shared::DType;

    using ThreadLayout = tile_layout::ColMajor<16, 2>;
    static constexpr int kThreadsPerRow = tl::num_rows<ThreadLayout>;
    static constexpr int kThreadsPerCol = tl::num_cols<ThreadLayout>;
    static constexpr int kWarpSize = 32;

    static constexpr int kNumPerAccess =
        traits::TraitsBase<DType>::kNumPerAccess;

    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kExecCount =
        BaseShape::kCols / (kNumPerAccess * kThreadsPerCol);

    static constexpr int kColStride = kThreadsPerCol * kNumPerAccess;

    using BaseTileSharedLayout = tl::SharedLayoutWrapper<Shared>::Layout;
    using BaseTileGlobalLayout =
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<Int<Global::kRowStride>, _1>>;

    using TiledCopy = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, DType>{},
        cute::Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
                     Stride<Int<kThreadsPerCol>, _1>>{},
        cute::Layout<Shape<_1, Int<kNumPerAccess>>>{}));

    using DataLayoutPerThread = cute::Layout<Shape<Int<kNumPerAccess>, _1>,
                                             Stride<_1, Int<kNumPerAccess>>>;

    DEVICE SharedToGlobalBaseTileStorer() : tiled_copy_(TiledCopy{}) {}

    DEVICE void copy(const DType* src, DType* dst) {
        int offset = 0;
#pragma unroll
        for (int i = 0; i < kExecCount; ++i) {
            auto src_tensor =
                make_tensor(make_smem_ptr(src + offset), data_layout_);
            auto dst_tensor =
                make_tensor(make_gmem_ptr(dst + offset), data_layout_);

            cute::copy(tiled_copy_, src_tensor, dst_tensor);

            offset += kColStride;
        }
    }

    DEVICE int lane_row_id() {
        int lane_id = threadIdx.x % warpSize;
        return lane_id % tl::num_rows<ThreadLayout>;
    }

    /// @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_col_id() {
        int lane_id = threadIdx.x % warpSize;
        return lane_id / tl::num_rows<ThreadLayout>;
    }

  private:
    DataLayoutPerThread data_layout_;
    TiledCopy tiled_copy_;
};

}  // namespace tiledcuda::cell::copy::atom
