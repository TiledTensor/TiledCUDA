#pragma once

#include "cell/traits/base.hpp"
#include "types/common.hpp"

namespace tiledcuda::cell {

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

template <class Element_, class TileLayout_>
class Shared {
  public:
    using DType = Element_;
    using Layout = TileLayout_;

    static constexpr int kNumel = tl::get_numel<Layout>;
    static constexpr int kRows = tl::num_rows<Layout>;
    static constexpr int kCols = tl::num_cols<Layout>;

    DEVICE Shared(DType* data) : data_(data) {}

  private:
    DType* data_;
};

}  // namespace tiledcuda::cell
