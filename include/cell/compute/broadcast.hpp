#pragma once

#include "cuda_utils.hpp"
#include "types/layout.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell::compute {
namespace tl = tile_layout;

namespace detail {

template <typename SrcTile, typename DstTile, const tl::Layout kLayout>
struct Broadcast {
    using DType = typename DstTile::DType::DType;

    static constexpr int kRows = DstTile::kRows;
    static constexpr int kCols = DstTile::kCols;

    DEVICE void operator()(const SrcTile& src, DstTile& dst) {}
};

template <typename SrcTile, typename DstTile>
struct Broadcast<SrcTile, DstTile, tl::Layout::kRowMajor> {
    using DType = typename DstTile::DType::DType;

    static constexpr int kRows = DstTile::kRows;
    static constexpr int kCols = DstTile::kCols;

    DEVICE void operator()(const SrcTile& src, DstTile& dst) {
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
            DType top_row = src(i, 0);
            DType bottom_row = src(i, 1);
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                dst(i, j)(0, 0) = top_row;
                dst(i, j)(0, 1) = top_row;
                dst(i, j)(1, 0) = top_row;
                dst(i, j)(1, 1) = top_row;

                dst(i, j)(0, 2) = bottom_row;
                dst(i, j)(0, 3) = bottom_row;
                dst(i, j)(1, 2) = bottom_row;
                dst(i, j)(1, 3) = bottom_row;
            }
        }
    }
};

}  // namespace detail
}  // namespace tiledcuda::cell::compute