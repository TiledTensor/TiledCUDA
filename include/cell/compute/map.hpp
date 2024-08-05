#pragma once

#include "cell/warp.hpp"
#include "cuda_utils.hpp"
#include "types/layout.hpp"

namespace tiledcuda::cell::compute {

namespace tl = tile_layout;

namespace detail {

template <typename RegTile, const tl::Layout kLayout>
struct Map {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename Functor>
    DEVICE void operator()(const RegTile& src, RegTile& dst, Functor functor) {}
};

template <typename RegTile>
struct Map<RegTile, tl::Layout::kRowMajor> {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename Functor>
    DEVICE void operator()(const RegTile& src, RegTile& dst, Functor functor) {
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                dst(i, j)(0, 0) = functor(src(i, j)(0, 0));
                dst(i, j)(0, 1) = functor(src(i, j)(0, 1));
                dst(i, j)(1, 0) = functor(src(i, j)(1, 0));
                dst(i, j)(1, 1) = functor(src(i, j)(1, 1));
                dst(i, j)(0, 2) = functor(src(i, j)(0, 2));
                dst(i, j)(0, 3) = functor(src(i, j)(0, 3));
                dst(i, j)(1, 2) = functor(src(i, j)(1, 2));
                dst(i, j)(1, 3) = functor(src(i, j)(1, 3));
            }
        }
    }
};

template <typename RegTile>
struct Map<RegTile, tl::Layout::kColMajor> {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename Functor>
    DEVICE void operator()(const RegTile& src, RegTile& dst, Functor functor) {
#pragma unroll
        for (int j = 0; j < kCols; ++j) {
#pragma unroll
            for (int i = 0; i < kRows; ++i) {
                dst(i, j)(0, 0) = functor(src(i, j)(0, 0));
                dst(i, j)(1, 0) = functor(src(i, j)(1, 0));
                dst(i, j)(0, 1) = functor(src(i, j)(0, 1));
                dst(i, j)(1, 1) = functor(src(i, j)(1, 1));
                dst(i, j)(0, 2) = functor(src(i, j)(2, 0));
                dst(i, j)(1, 2) = functor(src(i, j)(3, 0));
                dst(i, j)(0, 3) = functor(src(i, j)(2, 1));
                dst(i, j)(1, 3) = functor(src(i, j)(3, 1));
            }
        }
    }
};

}  // namespace detail
}  // namespace tiledcuda::cell::compute