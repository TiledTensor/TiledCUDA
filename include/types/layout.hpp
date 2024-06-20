#pragma once

#include "config.hpp"

#include <cute/layout.hpp>

namespace tiledcuda::cell {

/**
 * @namespace tile_layout
 *
 * @brief This namespace provides a set of utilities for defining tile layouts.
 * since Layout is quite common a name in various tensor libraries, we use
 * tile_layout to avoid potential name conflicts.
 */

namespace tile_layout {

using namespace cute;

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

template <typename Layout_>
static constexpr size_t num_rows = cute::size<0>(Layout_{});

template <typename Layout_>
static constexpr size_t num_cols = cute::size<1>(Layout_{});

template <typename Layout_>
static constexpr size_t row_stride = cute::size<0>(Layout_{}.layout().stride());

template <typename Layout_>
static constexpr size_t col_stride = cute::size<1>(Layout_{}.layout().stride());

template <typename Layout_>
static constexpr size_t get_numel = int(size(Layout_{}));

template <const int kShape1, const int kShape2, const int kStride1,
          const int kStride2>
HOST_DEVICE auto make_tile_layout() {
    printf("make_tile_layout\n");
    printf("kShape1 = %d, kShape2 = %d, kStride1 = %d, kStride2 = %d\n",
           kShape1, kShape2, kStride1, kStride2);

    using Layout = cute::Layout<Shape<Int<kShape1>, Int<kShape2>>,
                                Stride<Int<kStride1>, Int<kStride2>>>;
    return Layout{};
}

HOST_DEVICE auto make_row_major_layout(const int row, const int col,
                                       const int stride) {
    return cute::make_layout(make_shape(row, col),
                             make_stride(stride, Int<1>{}));
}

HOST_DEVICE auto make_col_major_layout(const int row, const int col,
                                       const int stride) {
    return cute::make_layout(make_shape(row, col),
                             make_stride(Int<1>{}, stride));
}

// CuTe's swizzle functions, swizzle(B, M, S), permute elements in a
// 2D coordinate space. This 2D coordinate space has 2^B rows and 2^S columns,
// and each coordinate position has 2^M elements. Therefore, to apply a swizzle
// function to a 2D data tile, the data tile should have a shape that is a
// multiple of 2^B x 2^S x 2^M.
/// @tparam Element: element type
/// @tparam kRows: number of rows
/// @tparam kCols: number of columns
/// @tparam kSwizzleMode: The value should be either 0 or 1, indicating whether
/// the size of the contiguous dimension is divisible by 32 or not.
template <typename Element, const int kRows, const int kCols,
          const int kSwizzleMode>
struct SwizzledRowMajor {};

// FIXME(haruhi): This implementation is very inflexible and is almost
// equivalent to a hard-coded swizzle function for 2D data tiles that have a
// shape that is a multiple of 8x32.
// Improve the implementation to make it more general.
template <const int kRows, const int kCols>
struct SwizzledRowMajor<cutlass::half_t, kRows, kCols, 0> {
    static_assert(kRows % 8 == 0,
                  "The number of rows must be a multiple of 8.");

    using SmemLayoutAtom = decltype(composition(
        Swizzle<2, 3, 3>{}, cute::Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
    using SmemLayout = decltype(tile_to_shape(SmemLayoutAtom{},
                                              Shape<Int<kRows>, Int<kCols>>{}));
};

// FIXME(haruhi): This implementation is very inflexible and is almost
// equivalent to a hard-coded swizzle function for 2D data tiles that have a
// shape that is a multiple of 8x64.
// Improve the implementation to make it more general.
template <const int kRows, const int kCols>
struct SwizzledRowMajor<cutlass::half_t, kRows, kCols, 1> {
    static_assert(kRows % 8 == 0,
                  "The number of rows must be a multiple of 8.");

    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{}, cute::Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
    using SmemLayout = decltype(tile_to_shape(SmemLayoutAtom{},
                                              Shape<Int<kRows>, Int<kCols>>{}));
};

template <const int kRows, const int kCols>
struct SwizzledRowMajor<float, kRows, kCols, 0> {
    // FIXME: not implmented yet. placeholder for future implementation.
    using SmemLayout = RowMajor<kRows, kCols, kCols>;
};

template <const int kRows, const int kCols>
struct SwizzledRowMajor<float, kRows, kCols, 1> {
    // FIXME: not implmented yet. placeholder for future implementation.
    using SmemLayout = RowMajor<kRows, kCols, kCols>;
};

}  // namespace tile_layout
}  // namespace tiledcuda::cell
