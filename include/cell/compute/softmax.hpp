#pragma once

#include "cell/compute/reduce.hpp"
#include "cuda_utils.hpp"
#include "types/layout.hpp"
#include "types/mod.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell::compute {

namespace tl = tile_layout;

template <typename RegTile, const tl::Layout kLayout>
struct Softmax {
    using DType = typename RegTile::DType::DType;
    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;
};

template <typename RegTile>
struct Softmax<RegTile, tl::Layout::kRowMajor> {
    using DType = typename RegTile::DType::DType;
    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename ReduceTile>
    DEVICE void operator()(RegTile& tile, ReduceTile& reduce_tile) {
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                auto base_tile = tile(i, j);

                tile(i, j)(0, 0) = exp(base_tile(0, 0));
                tile(i, j)(0, 1) = exp(base_tile(0, 1));
                tile(i, j)(1, 0) = exp(base_tile(1, 0));
                tile(i, j)(1, 1) = exp(base_tile(1, 1));
                tile(i, j)(0, 2) = exp(base_tile(0, 2));
                tile(i, j)(0, 3) = exp(base_tile(0, 3));
                tile(i, j)(1, 2) = exp(base_tile(1, 2));
                tile(i, j)(1, 3) = exp(base_tile(1, 3));
            }
        }

        compute::SumReduce<RegTile, tl::Layout::kRowMajor> row_sum;
        row_sum(tile, reduce_tile);

        // __syncthreads();

#pragma unroll
        for (int i = 0; i < kRows; ++i) {
            DType top_row_sum = reduce_tile(i, 0);
            DType bottom_row_sum = reduce_tile(i, 1);

#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                auto base_tile = tile(i, j);

                tile(i, j)(0, 0) = base_tile(0, 0) / top_row_sum;
                tile(i, j)(0, 1) = base_tile(0, 1) / top_row_sum;
                tile(i, j)(1, 0) = base_tile(1, 0) / top_row_sum;
                tile(i, j)(1, 1) = base_tile(1, 1) / top_row_sum;

                tile(i, j)(0, 2) = base_tile(0, 2) / bottom_row_sum;
                tile(i, j)(0, 3) = base_tile(0, 3) / bottom_row_sum;
                tile(i, j)(1, 2) = base_tile(1, 2) / bottom_row_sum;
                tile(i, j)(1, 3) = base_tile(1, 3) / bottom_row_sum;
            }
        }
    }
};

template <typename RegTile>
struct Softmax<RegTile, tl::Layout::kColMajor> {
    using DType = typename RegTile::DType::DType;
    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;
};

}  // namespace tiledcuda::cell::compute
