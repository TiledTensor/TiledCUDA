#pragma once

#include "cell/copy/constants.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

namespace details {

template <typename Element, typename Reg, const tl::Layout type>
struct LoadTileImpl {
    using DType = Element;

    DEVICE void operator()(const DType* src, Reg& dst, const int stride);
};

/// @brief Load tile from global memory to register tile in RowMajor layout.
/// @tparam Element Data type of the elements.
/// @tparam Reg Register tile type.
template <typename Element, typename Reg>
struct LoadTileImpl<Element, Reg, tl::Layout::RowMajor> {
    using DType = Element;
    static constexpr int Height = Reg::kRows;
    static constexpr int Width = Reg::kCols;
    static constexpr int SubTileSize = 16;

    DEVICE void operator()(const DType* src, Reg& dst, const int stride) {
        int lane_id = threadIdx.x % warpSize;
        int warp_id = threadIdx.x / warpSize;
        const int tile_size = SubTileSize;
        int rows = Height * SubTileSize;
        int row_offset = rows * warp_id;

        // Load data from global memory to register.
        // References:
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=ldmatrix#warp-level-matrix-load-instruction-ldmatrix
#pragma unroll
        for (int i = 0; i < Height; ++i) {
            int row = row_offset + i * tile_size + (lane_id / 4);
#pragma unroll
            for (int j = 0; j < Width; ++j) {
                int col = j * tile_size + (lane_id % 4);
                dst(i, j)(0, 0) = src[(row + 0) * stride + col + 0];
                dst(i, j)(0, 1) = src[(row + 0) * stride + col + 1];
                dst(i, j)(1, 0) = src[(row + 0) * stride + col + 8];
                dst(i, j)(1, 1) = src[(row + 0) * stride + col + 9];
            }
#pragma unroll
            for (int j = 0; j < Width; ++j) {
                int col = j * tile_size + (lane_id % 4);
                dst(i, j)(0, 2) = src[(row + 8) * stride + col + 0];
                dst(i, j)(0, 3) = src[(row + 8) * stride + col + 1];
                dst(i, j)(1, 2) = src[(row + 8) * stride + col + 8];
                dst(i, j)(1, 3) = src[(row + 8) * stride + col + 9];
            }
        }
    }
};

/// @brief Load tile from global memory to register tile in ColMajor layout.
/// @tparam Element Data type of the elements.
/// @tparam Reg Register tile type.
// TODO(KuangjuX): Implement LoadImpl for ColMajor layout.
template <typename Element, typename Reg>
struct LoadTileImpl<Element, Reg, tl::Layout::ColMajor> {};

}  // namespace details

template <typename Element, typename Reg_, tl::Layout type_>
struct GlobalToRegLoader {
    using Reg = Reg_;
    using DType = Element;

    DEVICE void operator()(const DType* src, Reg& dst, const int stride) {
        details::LoadTileImpl<DType, Reg, type_> loader;
        loader(src, dst, stride);
    }
};

}  // namespace tiledcuda::cell::copy
