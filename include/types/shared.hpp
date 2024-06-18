#pragma once

#include "cell/traits/base.hpp"
#include "types/common.hpp"
#include "types/layout.hpp"

namespace tiledcuda::cell {

using namespace cell;
namespace tl = tile_layout;

namespace {
template <typename Element, const int kRows, const int kCols, const int kStride>
__device__ void print_tile(const Element* data) {
    const int delimeter = 64;

    half* data_ = reinterpret_cast<half*>(const_cast<Element*>(data));

    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    int row, col;
    for (int i = 0; i < kRows * kCols; ++i) {
        row = i / kCols;
        col = i % kCols;

        printf("%.0f, ", __half2float(data_[row * kStride + col]));

        if ((i + 1) % delimeter == 0) printf("\n");
    }
    printf("\n\n");
}
}  // namespace

template <class Element_, class Layout_>
class SharedTile {
  public:
    using DType = Element_;
    using Layout = Layout_;

    static constexpr int kNumel = tl::get_numel<Layout>;
    static constexpr int kRows = tl::num_rows<Layout>;
    static constexpr int kCols = tl::num_cols<Layout>;

    DEVICE SharedTile(DType* data) : data_(data), layout_(Layout{}) {}

    // for write access
    DEVICE DType& operator()(int x, int y) { return data_[layout_(x, y)]; }

    // for read access
    DEVICE const DType& operator()(int x, int y) const {
        return data_[layout_(x, y)];
    }

  private:
    DType* data_;
    Layout layout_;
};

}  // namespace tiledcuda::cell
