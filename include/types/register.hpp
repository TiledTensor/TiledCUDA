#pragma once

#include "types/layout.hpp"
#include "util/print.hpp"

namespace tiledcuda::cell {

namespace tl = tile_layout;

template <typename Element_, typename Layout_>
class RegTile {
  public:
    using DType = Element_;
    using Layout = Layout_;

    static constexpr int kNumel = tl::get_numel<Layout>;
    static constexpr int kRows = tl::num_rows<Layout>;
    static constexpr int kCols = tl::num_cols<Layout>;

    static constexpr tl::Layout kType = tl::layout_type<Layout>;

    DEVICE RegTile() : layout_(Layout{}) {
        memset((void*)data_, 0, sizeof(data_));
    }

    DEVICE DType* mutable_data() { return (DType*)data_; }

    DEVICE const DType* data() const { return (DType*)data_; }

    DEVICE const Layout& layout() const { return layout_; }

    // for write access
    DEVICE DType& operator()(int x, int y) { return data_[layout_(x, y)]; }

    // for read access
    DEVICE const DType& operator()(int x, int y) const {
        return data_[layout_(x, y)];
    }

    DEVICE void dump_value() {
        print_tile(static_cast<DType*>(data_), layout_);
    }

  private:
    DType data_[kNumel];
    Layout layout_;
};

template <typename Element>
using BaseTileRowMajor = RegTile<Element, tl::RowMajor<2, 4>>;

template <typename Element>
using BaseTileColMajor = RegTile<Element, tl::ColMajor<4, 2>>;
}  // namespace tiledcuda::cell
