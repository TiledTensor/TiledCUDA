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

enum class Layout {
    kRowMajor = 0,  // Tile layout for shared memory.
    kColMajor = 1,
    kSwizzledRowMajor = 2,
    kSwizzledColMajor = 3
};

// In the row major layout, the contiguous dimension in memory is the
// last dimension.
template <const int kRow, const int kCol, const int kStride = kCol>
using RowMajor =
    cute::Layout<Shape<Int<kRow>, Int<kCol>>, Stride<Int<kStride>, _1>>;

// In the column major layout, the contiguous dimension in memory is the
// first dimension.
template <const int kRow, const int kCol, const int kStride = kRow>
using ColMajor =
    cute::Layout<Shape<Int<kRow>, Int<kCol>>, Stride<_1, Int<kStride>>>;

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

// We wrap CuTe's `Layout`, which consists of `Shape` and `Stride`, into an
// intelligent row-major or column-major layout. In a row-major layout, the
// column stride is 1, whereas in a column-major layout, the row stride is 1.
// NOTE: A potential issue is that `ColMajor<1, 1>` will also be indentified as
// a row-major layout.
template <typename Layout_>
static constexpr Layout layout_type =
    col_stride<Layout_> == 1 ? Layout::kRowMajor : Layout::kColMajor;

template <const int kShape1, const int kShape2, const int kStride1,
          const int kStride2>
HOST_DEVICE auto make_tile_layout() {
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

// @brief: leverage CuTe's swizzle functions, swizzle(B, M, S), permute elements
//         in a 2D coordinate space. This 2D coordinate space has 2^B rows and
//         2^S columns, and each coordinate position has 2^M elements.
//         Therefore, to apply a swizzle function to a 2D data tile, the data
//         tile should have a shape that is a multiple of 2^B x 2^S x 2^M.
template <typename Layout_, const int kB, const int kM, const int kS>
struct Swizzled {
    static_assert(int(cute::size(Layout_{})) % 2 ^ kB * 2 ^ kM * 2 ^ kS == 0);

    using LayoutAtom =
        decltype(composition(cute::Swizzle<kB, kS, kM>{},
                             cute::Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
    using Layout = decltype(tile_to_shape(
        LayoutAtom{}, cute::Shape<Int<cute::size<0>(Layout_{})>,
                                  Int<cute::size<1>(Layout_{})>>{}));
};

}  // namespace tile_layout
}  // namespace tiledcuda::cell
