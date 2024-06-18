#pragma once

#include "cell/traits/base.hpp"
#include "types/layout.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell {

namespace tl = tile_layout;

template <class Element_, class Layout_>
class RegTile {
  public:
    using DType = Element_;
    using Layout = Layout_;

    static constexpr int kNumel = tl::get_numel<Layout>;
    static constexpr int kRows = tl::num_rows<Layout>;
    static constexpr int kCols = tl::num_cols<Layout>;

    DEVICE RegTile() : layout_(Layout{}) {
        memset((void*)data_, 0, sizeof(data_));
    }

    // for write access
    DEVICE DType& operator()(int x, int y) { return data_[layout_(x, y)]; }

    // for read access
    DEVICE const DType& operator()(int x, int y) const {
        return data_[layout_(x, y)];
    }

  private:
    DType data_[kNumel];
    Layout layout_;
};

}  // namespace tiledcuda::cell
