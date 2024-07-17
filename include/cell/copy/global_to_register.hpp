#pragma once

#include "cell/copy/constants.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

namespace details {

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
struct GlobalToRegLoaderImpl<Global, Reg, tl::Layout::ColMajor> {
    using DType = typename Global::DType;
    static constexpr int kHeight = Reg::kRows;
    static constexpr int kWidth = Reg::kCols;
    static constexpr int kSubTileSize = 16;

    DEVICE void operator()(const Global& src, Reg& dst) {}
};

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

}  // namespace tiledcuda::cell::copy
