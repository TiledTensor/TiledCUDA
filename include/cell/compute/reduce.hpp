#pragma once

#include "cell/traits/base.hpp"
#include "cuda_utils.hpp"
#include "types/layout.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell::compute {

namespace tl = tile_layout;

namespace detail {

template <typename RegTile>
struct Reduce {
    using DType = typename RegTile::DType::DType;
    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename Reduce>
    DEVICE operator()(RegTile& tile, Reduce reduce) {
        // TODO: Implement the reduction logic.
    }

  private:
    template <typename Reduce>
    DEVICE DType warp_reduce(DType data, unsigned mask, Reduce reduce) {
#pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            data = reduce(data, __shfl_down_sync(mask, data, offset, 32));
        }

        return data;
    }
};
}  // namespace detail
}  // namespace tiledcuda::cell::compute