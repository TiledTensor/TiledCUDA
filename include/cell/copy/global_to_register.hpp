#pragma once

#include "cell/copy/constants.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

namespace details {

template <typename Element, typename Reg, const tl::Layout type>
struct GlobalToRegLoaderImpl {
    using DType = Element;
    static constexpr int kHeight = Reg::kRows;
    static constexpr int kWidth = Reg::kCols;
    static constexpr int kSubTileSize = 16;

    DEVICE void operator()(const DType* src, Reg& dst, const int stride);
};

/// @brief Load tile from global memory to register tile in RowMajor layout.
/// @tparam Element Data type of the elements.
/// @tparam Reg Register tile type.
template <typename Element, typename Reg>
struct GlobalToRegLoaderImpl<Element, Reg, tl::Layout::RowMajor> {
    using DType = Element;
    static constexpr int kHeight = Reg::kRows;
    static constexpr int kWidth = Reg::kCols;
    static constexpr int kSubTileSize = 16;

    DEVICE void operator()(const DType* src, Reg& dst, const int stride) {
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
                dst(i, j)(0, 0) = src[(row + 0) * stride + col + 0];
                dst(i, j)(0, 1) = src[(row + 0) * stride + col + 1];
                dst(i, j)(1, 0) = src[(row + 0) * stride + col + 8];
                dst(i, j)(1, 1) = src[(row + 0) * stride + col + 9];
            }
#pragma unroll
            for (int j = 0; j < kWidth; ++j) {
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
struct GlobalToRegLoaderImpl<Element, Reg, tl::Layout::ColMajor> {};

}  // namespace details

template <typename Element, typename Reg_, tl::Layout type_>
struct GlobalToRegLoader {
    using Reg = Reg_;
    using DType = Element;

    DEVICE void operator()(const DType* src, Reg& dst, const int stride) {
        details::GlobalToRegLoaderImpl<DType, Reg, type_> loader;
        loader(src, dst, stride);
    }
};

}  // namespace tiledcuda::cell::copy
