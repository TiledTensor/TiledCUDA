#pragma once

#include "cell/copy/constants.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

template <typename T, typename Layout>
struct GlobalToRegLoader {
    DEVICE void operator()(const T* global, RegTile<T, Layout>& dst);
}

template <typename T, size_t height, size_t width>
DEVICE void load_global_to_reg(
    T* src,
    RegTile<RegTile<T, tl::RowMajor<2, 4>>, tl::RowMajor<height, width>>& dst,
    const int row_stride) {
    // TODO: Fix constants.
    const int WARP_SIZE = 32;
    const int sub_tile_size = 16;
    const int sub_rows = sub_tile_size;
    const int sub_cols = sub_tile_size;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int rows = height * sub_rows;
    int cols = width * sub_cols;
    int sub_tile = sub_tile_size;
    int row_offset = rows * warp_id;

    // RegTile<T, tl::RowMajor<2, 4>> subtile_reg;
    // RegTile<RegTile<T, tl::RowMajor<2, 4>>, tl::RowMajor<height, width>>
    //     dst_reg;

// Load data from global memory to register.
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

}  // namespace tiledcuda::cell::copy