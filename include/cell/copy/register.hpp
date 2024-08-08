#pragma once

#include "cuda_utils.hpp"

namespace tiledcuda::cell::copy {

template <typename Element>
struct DataCopy {
    DEVICE void operator()(const Element& src, Element& dst) { dst = src; }
};

template <typename RegTile, typename Copy>
struct RegCopy {
    using DType = typename RegTile::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    DEVICE void operator()(const RegTile& src, RegTile& dst) {
        Copy c;
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                c(src(i, j), dst(i, j));
            }
        }
    }
};

template <typename RegTile>
using BaseTileCopy = RegCopy<RegTile, DataCopy<typename RegTile::DType>>;
template <typename RegTile>
using RegTileCopy = RegCopy<RegTile, BaseTileCopy<typename RegTile::DType>>;

}  // namespace tiledcuda::cell::copy
