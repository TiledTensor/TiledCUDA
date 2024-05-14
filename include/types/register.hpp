#pragma once

#include "config.hpp"
#include "types/layout.hpp"

namespace tiledcuda {

namespace tl = tile_layout;

template <typename Element_, typename Layout_>
struct RegTile {
    using Element = Element_;

    constexpr static int kRows = tl::num_rows<Layout_>;
    constexpr static int kCols = tl::num_cols<Layout_>;

    DEVICE Element* operator[](size_t idx) { return &data_[idx][0]; }

    DEVICE const Element* operator[](size_t idx) const {
        return &data_[idx][0];
    }

    DEVICE Element& operator[](int2 idx) { return data_[idx.x][idx.y]; }

    DEVICE const Element& operator[](int2 idx) const {
        return data_[idx.x][idx.y];
    }

    Element data_[kRows][kCols];  // register tile data
};

}  // namespace tiledcuda
