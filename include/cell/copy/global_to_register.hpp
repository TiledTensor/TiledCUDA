#pragma once

#include "cell/copy/constants.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

/**
 * @brief Copy a 2D tile from global memory to register tile
 * @param src[in] source data in global memory.
 * @param dst[out] destination data in register tile.
 * @param row_stride[in] row stride of source data.
 */
template <typename T, typename G2RTraits, size_t height, size_t width>
DEVICE void copy_2d_tile_g2r(
    const T* src,
    RegTile<RegTile<T, tl::RowMajor<2, 4>>, tl::RowMajor<height, width>>& dst,
    const int row_stride) {
    static constexpr int WARP_SIZE = G2RTraits::WARP_SIZE;
    const int sub_tile_size = G2RTraits::SubTileSize;
    const int sub_rows = sub_tile_size;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int rows = height * sub_rows;
    int tile_size = sub_tile_size;
    int row_offset = rows * warp_id;

    switch (G2RTraits::GLOBAL_LAYOUT) {
        case tile_layout::GlobalLayout::RowMajor: {
            // Load data from global memory to register.
            // References:
            // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=ldmatrix#warp-level-matrix-load-instruction-ldmatrix
#pragma unroll
            for (int i = 0; i < height; ++i) {
                int row = row_offset + i * tile_size + (lane_id / 4);
#pragma unroll
                for (int j = 0; j < width; ++j) {
                    int col = j * tile_size + (lane_id % 4);
                    dst(i, j)(0, 0) = src[(row + 0) * row_stride + col + 0];
                    dst(i, j)(0, 1) = src[(row + 0) * row_stride + col + 1];
                    dst(i, j)(1, 0) = src[(row + 0) * row_stride + col + 8];
                    dst(i, j)(1, 1) = src[(row + 0) * row_stride + col + 9];
                }
#pragma unroll
                for (int j = 0; j < width; ++j) {
                    int col = j * tile_size + (lane_id % 4);
                    dst(i, j)(0, 2) = src[(row + 8) * row_stride + col + 0];
                    dst(i, j)(0, 3) = src[(row + 8) * row_stride + col + 1];
                    dst(i, j)(1, 2) = src[(row + 8) * row_stride + col + 8];
                    dst(i, j)(1, 3) = src[(row + 8) * row_stride + col + 9];
                }
            }
        }
        case tile_layout::GlobalLayout::ColMajor: {
            // TODO: Add support for ColMajor layout.
            break;
        }
    }
}

namespace details {

template <typename Element, typename Reg, const tl::Layout type>
struct LoadTileImpl {
    using DType = Element;

    DEVICE void operator()(const DType* src, Reg& dst, const int stride);
};

template <typename Element, typename Reg>
struct LoadTileImpl<Element, Reg, tl::Layout::RowMajor> {
    using DType = Element;
    static constexpr int Height = Reg::kRows;
    static constexpr int Width = Reg::kCols;
    static constexpr int SUB_TILE_SIZE = 16;

    DEVICE void operator()(const DType* src, Reg& dst, const int stride) {
        int lane_id = threadIdx.x % warpSize;
        int warp_id = threadIdx.x / warpSize;
        const int tile_size = SUB_TILE_SIZE;
        int rows = Height * SUB_TILE_SIZE;
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
