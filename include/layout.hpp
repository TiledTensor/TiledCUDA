#pragma once

#include "config.hpp"

#include <cute/layout.hpp>

using namespace cute;

namespace tiledcuda {
// In the row major layout, the contiguous dimension in memory is the
// last dimension.
template <const int row, const int col, const int stride = col>
using RowMajor =
    cute::Layout<Shape<Int<row>, Int<col>>, Stride<Int<stride>, _1>>;

// In the column major layout, the contiguous dimension in memory is the
// first dimension.
template <const int row, const int col, const int stride = row>
using ColMajor =
    cute::Layout<Shape<Int<row>, Int<col>>, Stride<_1, Int<stride>>>;

HOST_DEVICE auto make_row_major_layout(const int row, const int col,
                                       const int stride) {
    return cute::make_layout(make_shape(row, col), make_stride(stride, 1));
}

HOST_DEVICE auto make_col_major_layout(const int row, const int col,
                                       const int stride) {
    return cute::make_layout(make_shape(row, col), make_stride(1, stride));
}
}  // namespace tiledcuda
