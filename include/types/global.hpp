#pragma once

#include "types/layout.hpp"
#include "util/print.hpp"

namespace tiledcuda::cell {

namespace tl = tile_layout;

template <typename Element_, typename Layout_>
struct GlobalTile {
    using DType = Element_;
    using Layout = Layout_;

    static constexpr int kNumel = tl::get_numel<Layout>;

    static constexpr int kRows = tl::num_rows<Layout>;
    static constexpr int kCols = tl::num_cols<Layout>;

    static constexpr int kRowStride = tl::row_stride<Layout>;
    static constexpr int kColStride = tl::col_stride<Layout>;

    static constexpr tl::Layout kType = tl::layout_type<Layout>;

    DEVICE GlobalTile(DType* data) : data_(data), layout_(Layout{}) {}

    DEVICE DType* mutable_data() { return data_; }

    DEVICE const DType* data() const { return data_; }

    HOST_DEVICE const Layout& layout() const { return layout_; }

    // for write access
    DEVICE DType& operator()(int x, int y) { return data_[layout_(x, y)]; }

    // for read access
    DEVICE
    const DType& operator()(int x, int y) const { return data_[layout_(x, y)]; }

    DEVICE void dump_value() { print_tile(data_, layout_); }

  private:
    DType* data_;
    Layout layout_;
};
}  // namespace tiledcuda::cell
