#pragma once

#include "config.hpp"
#include "types/layout.hpp"

namespace tiledcuda {

namespace tl = tile_layout;

template <typename Element_, typename Layout_>
class RegTile {
  public:
    using Element = Element_;
    using Layout = Layout_;  // to keep the behavior consistent with SharedTile

    constexpr static int kRows = tl::num_rows<Layout_>;
    constexpr static int kCols = tl::num_cols<Layout_>;

    DEVICE Element* operator[](size_t idx) { return &data_[idx][0]; }

    DEVICE const Element* operator[](size_t idx) const {
        return &data_[idx][0];
    }

    DEVICE Element* operator[](int2 idx) { return &data_[idx.x][idx.y]; }

    DEVICE const Element* operator[](int2 idx) const {
        return &data_[idx.x][idx.y];
    }

  private:
    Element data_[kRows][kCols];  // register tile data
};

}  // namespace tiledcuda
