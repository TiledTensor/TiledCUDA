#pragma once

#include "cell/copy/constants.hpp"
#include "cell/copy/warp.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

using namespace tiledcuda::cell::traits;

namespace details {

// template <typename Element, const tl::Layout type>
// struct LoadBaseTile {
//     using DType = Element;

//     DEVICE void operator()(const DType* src, BaseTileRowMajor<DType>& dst) {
//         int lane_id = threadIdx.x % warpSize;
//     }
// };

// template <typename Element>
// struct LoadBaseTile<Element, tl::Layout::RowMajor> {
//     using DType = Element;
//     using BaseTile = BaseTileRowMajor<Element>;
// };

template <typename Global, typename Reg, const tl::Layout type>
struct GlobalToRegLoaderImpl {
    using DType = Global::type;
    static constexpr int kHeight = Reg::kRows;
    static constexpr int kWidth = Reg::kCols;
    static constexpr int kSubTileSize = 16;

    DEVICE void operator()(const Global& src, Reg& dst);
};

/// @brief Load tile from global memory to register tile in RowMajor layout.
/// @tparam Global Global tile type.
/// @tparam Reg Register tile type.
template <typename Global, typename Reg>
struct GlobalToRegLoaderImpl<Global, Reg, tl::Layout::RowMajor> {
    using DType = typename Global::DType;
    static constexpr int kHeight = Reg::kRows;
    static constexpr int kWidth = Reg::kCols;
    static constexpr int kSubTileSize = 16;

    DEVICE void operator()(const Global& src, Reg& dst) {
        int lane_id = threadIdx.x % warpSize;
        int warp_id = threadIdx.x / warpSize;
        const int tile_size = kSubTileSize;
        int rows = kHeight * kSubTileSize;
        int row_offset = rows * warp_id;

        // Load data from global memory to register.
        // References:
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=ldmatrix#warp-level-matrix-load-instruction-ldmatrix
#pragma unroll
        for (int i = 0; i < kHeight; ++i) {
            int row = row_offset + i * tile_size + (lane_id / 4);
#pragma unroll
            for (int j = 0; j < kWidth; ++j) {
                int col = j * tile_size + (lane_id % 4);
                dst(i, j)(0, 0) = src(row + 0, col + 0);
                dst(i, j)(0, 1) = src(row + 0, col + 1);
                dst(i, j)(1, 0) = src(row + 0, col + 8);
                dst(i, j)(1, 1) = src(row + 0, col + 9);
            }
#pragma unroll
            for (int j = 0; j < kWidth; ++j) {
                int col = j * tile_size + (lane_id % 4);
                dst(i, j)(0, 2) = src(row + 8, col + 0);
                dst(i, j)(0, 3) = src(row + 8, col + 1);
                dst(i, j)(1, 2) = src(row + 8, col + 8);
                dst(i, j)(1, 3) = src(row + 8, col + 9);
            }
        }
    }
};

/// @brief Load tile from global memory to register tile in ColMajor layout.
/// @tparam Global Global tile type.
/// @tparam Reg Register tile type.
// TODO(KuangjuX): Implement LoadImpl for ColMajor layout.
template <typename Global, typename Reg>
struct GlobalToRegLoaderImpl<Global, Reg, tl::Layout::ColMajor> {};

}  // namespace details

template <typename Global_, typename Reg_, tl::Layout type_>
struct GlobalToRegLoader {
    using Global = Global_;
    using Reg = Reg_;
    using DType = typename Global::DType;

    DEVICE void operator()(const Global& src, Reg& dst) {
        details::GlobalToRegLoaderImpl<Global, Reg, type_> loader;
        loader(src, dst);
    }
};

template <typename Global_, typename Reg_, const int kRowExec_,
          const int kColExec_, const tl::Layout type>
struct GlobalToRegWarpLoaderImpl {
    using Global = Global_;
    using Reg = Reg_;
    using DType = Global::type;

    DEVICE void operator()(const DType* src, Reg& dst);
};

template <typename Global_, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct GlobalToRegWarpLoaderImpl<Global_, Reg_, kRowExec_, kColExec_,
                                 tl::Layout::RowMajor> {
    using Global = Global_;
    using Reg = Reg_;
    using DType = Global::DType;

    // strides to iterate over each 16x128-bits `BaseTile`.
    static constexpr int kTileRstride =
        BaseTileShape<DType>::row * Global::kRowStride;
    static constexpr int kTileCstride = BaseTileShape<DType>::col;

    // row stride and col stride for a thread in a warp
    static constexpr int kStride = traits::TraitsBase<DType>::kNumPerAccess;
    static constexpr int kLaneRstride = Global::kRowStride;
    static constexpr int kLaneCstride = kStride;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const DType* src, Reg& dst) {
        int lane_id = threadIdx.x % warpSize;

        const DType* data = src;
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
            // TODO: magic number.
            int row = i * 16 + lane_id / 4;
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                // TODO: magic number.
                int col = j * 16 + lane_id % 4;
                dst(i, j)(0, 0) =
                    data[(row + 0) * Global::kRowStride + col + 0];
                dst(i, j)(0, 1) =
                    data[(row + 0) * Global::kRowStride + col + 1];
                dst(i, j)(1, 0) =
                    data[(row + 0) * Global::kRowStride + col + 8];
                dst(i, j)(1, 1) =
                    data[(row + 0) * Global::kRowStride + col + 9];
                dst(i, j)(0, 2) =
                    data[(row + 8) * Global::kRowStride + col + 0];
                dst(i, j)(0, 3) =
                    data[(row + 8) * Global::kRowStride + col + 1];
                dst(i, j)(1, 2) =
                    data[(row + 8) * Global::kRowStride + col + 8];
                dst(i, j)(1, 3) =
                    data[(row + 8) * Global::kRowStride + col + 9];
            }
        }
    }
};

template <typename Global_, typename Reg_, typename WarpLayout_,
          const WarpReuse kMode_,
          typename Base = warp::CopyBase<WarpLayout_, kMode_>>
struct GlobalToRegWarpLoader : public Base {
    using Global = Global_;
    using Reg = Reg_;
    using DType = typename Global::DType;

    using WarpLayout = WarpLayout_;
    static constexpr WarpReuse kMode = kMode_;

    DEVICE void operator()(const Global& src, Reg& dst) {
        const DType* src_ptr = src.data();

        // 1. advance the pointer to input data to the current warp according to
        // warp reuse mode.
        src_ptr += Base::template get_warp_offset<Global>();

        // how many times a `BaseTile` is executed along the row and column
        // direction.
        static constexpr int kRowExec =
            Base::template row_exec_count<DType, Global::kRows>();
        static constexpr int kColExec =
            Base::template col_exec_count<DType, Global::kCols>();

        // TODO: Load data from global memory to register.
        using Loader = GlobalToRegWarpLoaderImpl<Global, Reg, kRowExec,
                                                 kColExec, Global::type>;
        Loader loader;
        loader(src_ptr, dst);
    }
};

}  // namespace tiledcuda::cell::copy
