#pragma once

#include "cell/compute/base.hpp"
#include "cell/traits/base.hpp"
#include "cuda_utils.hpp"
#include "types/layout.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell::compute {

namespace tl = tile_layout;

namespace detail {

template <typename RegTile, const tl::Layout kLayout>
struct Reduce {
    using DType = typename RegTile::DType::DType;
    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;
    static constexpr uint32_t kMaskAll = 0xFFFFFFFF;

    template <typename DstTile, typename Reduce>
    DEVICE void operator()(const RegTile& src, DstTile& dst, Reduce reduce) {}
};

template <typename RegTile>
struct Reduce<RegTile, tl::Layout::kRowMajor> {
    using DType = typename RegTile::DType::DType;
    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;
    static constexpr uint32_t kMaskAll = 0xFFFFFFFF;

    template <typename DstTile, typename Reduce>
    DEVICE void operator()(const RegTile& src, DstTile& dst, Reduce reduce) {
        const int leader = threadIdx.x & 0x1C;
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
            DType top_rows[kCols];
            DType bottom_rows[kCols];
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                auto base_tile = src(i, j);
                DType top_row_0 = reduce(base_tile(0, 0), base_tile(0, 1));
                DType top_row_1 = reduce(base_tile(1, 0), base_tile(1, 1));
                top_rows[j] = reduce(top_row_0, top_row_1);

                DType bottom_row_0 = reduce(base_tile(0, 2), base_tile(0, 3));
                DType bottom_row_1 = reduce(base_tile(1, 2), base_tile(1, 3));
                bottom_rows[j] = reduce(bottom_row_0, bottom_row_1);
            }

            DType top_row = top_rows[0];
            DType bottom_row = bottom_rows[0];

            // Compute the reduction of the top and bottom rows.
            for (int j = 1; j < kCols; ++j) {
                top_row = reduce(top_row, top_rows[j]);
                bottom_row = reduce(bottom_row, bottom_rows[j]);
            }

            // Shuffle the results to the leader thread.
            top_row = shuffle_down_sync(kMaskAll, top_row, 2);
            top_row = shuffle_down_sync(kMaskAll, top_row, 1);

            bottom_row = shuffle_down_sync(kMaskAll, bottom_row, 2);
            bottom_row = shuffle_down_sync(kMaskAll, bottom_row, 1);

            top_row = shuffle_down_sync(kMaskAll, top_row, leader);
            bottom_row = shuffle_down_sync(kMaskAll, bottom_row, leader);

            // Store the results to the destination tile.
            dst(i, 0) = top_row;
            dst(i, 1) = bottom_row;
        }
    }

  private:
    DEVICE DType shuffle_down_sync(uint32_t mask, DType value, int delta) {
        return __shfl_down_sync(mask, value, delta);
    }
};

template <typename RegTile>
struct Reduce<RegTile, tl::Layout::kColMajor> {
    using DType = typename RegTile::DType::DType;
    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename DstTile, typename Reduce>
    DEVICE void operator()(const RegTile& tile, DstTile& dst, Reduce reduce) {}
};

}  // namespace detail

template <typename RegTile, const tl::Layout kLayout>
struct SumReduce {
    using DType = typename RegTile::DType::DType;

    template <typename DstTile>
    DEVICE void operator()(const RegTile& src, DstTile& dst) {
        detail::Reduce<RegTile, kLayout> row_sum;
        row_sum(src, dst, Add<DType>{});
    }
};

template <typename RegTile, const tl::Layout kLayout>
struct MaxReduce {
    using DType = typename RegTile::DType::DType;

    template <typename DstTile>
    DEVICE void operator()(const RegTile& src, DstTile& dst) {
        detail::Reduce<RegTile, kLayout> row_max;
        row_max(src, dst, Max<DType>{});
    }
};

}  // namespace tiledcuda::cell::compute
