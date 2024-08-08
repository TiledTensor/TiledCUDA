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

template <const int kRows_, const int kCols_, const int kRowStride_,
          const int kColStride_>
struct MatrixLayout {
    static constexpr int kRows = kRows_;
    static constexpr int kCols = kCols_;

    static constexpr int kRowStride = kRowStride_;
    static constexpr int kColStride = kColStride_;

    static constexpr int kNumel = kRows * kCols;

    static constexpr Layout kType =
        kColStride == 1 ? Layout::kRowMajor : Layout::kColMajor;

    // NOTE: kSwizzled, kB, kM, and kS are used to maintain compatibility with
    // swizzled layouts.
    // In non-swizzled layouts, they are set to -1 by default.
    static constexpr bool kSwizzled = false;
    static constexpr int kB = -1;
    static constexpr int kM = -1;
    static constexpr int kS = -1;

    // FIXME: The current implementation for copying global to shared memory
    // relies on CuTe's API. This is a temporary workaround to maintain
    // compatibility with CuTe's APIs. Refactor this implementation in the
    // future to remove this dependency.
    using CuteLayout = cute::Layout<Shape<Int<kRows>, Int<kCols>>,
                                    Stride<Int<kRowStride>, Int<kColStride>>>;

    DEVICE int operator()(int i, int j) const {
        return i * kRowStride + j * kColStride;
    }
};

// In the row major layout, the contiguous dimension in memory is the
// last dimension.
template <const int kRow, const int kCol, const int kStride = kCol>
using RowMajor = MatrixLayout<kRow, kCol, kStride, 1>;

// In the column major layout, the contiguous dimension in memory is the
// first dimension.
template <const int kRow, const int kCol, const int kStride = kRow>
using ColMajor = MatrixLayout<kRow, kCol, 1, kStride>;

// @brief: leverage CuTe's swizzle functions, swizzle(B, M, S), permute elements
//         in a 2D coordinate space. This 2D coordinate space has 2^B rows and
//         2^S columns, and each coordinate position has 2^M elements.
//         Therefore, to apply a swizzle function to a 2D data tile, the data
//         tile should have a shape that is a multiple of 2^B x 2^S x 2^M.
template <typename Layout_, const int kB_, const int kM_, const int kS_>
struct Swizzled {
    static_assert(
        Layout_::kNumel % ((1 << kB_) * (1 << kM_) * (1 << kS_)) == 0,
        "The shape of the data tile should be a multiple of 2^B x 2^S x 2^M.");

    static constexpr int kRows = Layout_::kRows;
    static constexpr int kCols = Layout_::kCols;

    static constexpr int kRowStride = Layout_::kRowStride;
    static constexpr int kColStride = Layout_::kColStride;

    static constexpr int kNumel = Layout_::kNumel;

    // NOTE: Swizzling can be considered as either a row-major or column-major
    // layout. Since it does not affect subsequent load or store operations and
    // to avoid repeated code, we set the layout type to be the same as the
    // original layout.
    static constexpr Layout kType = Layout::kRowMajor;

    static constexpr bool kSwizzled = true;
    static constexpr int kB = kB_;
    static constexpr int kM = kM_;
    static constexpr int kS = kS_;

    // FIXME(haruhi): this should be tested on swizzled column major layout.
    static constexpr int kSwizzledRow = 1 << kS;
    static constexpr int kSwizzledCol = (1 << kB) * (1 << kM);

    static constexpr int kSwizzledRowStride =
        kType == Layout::kRowMajor ? kSwizzledCol : 1;
    static constexpr int kSwizzledColStride =
        kType == Layout::kRowMajor ? 1 : kSwizzledRow;

    using LayoutAtom = decltype(composition(
        cute::Swizzle<kB, kM, kS>{},
        cute::Layout<
            Shape<Int<kSwizzledRow>, Int<kSwizzledCol>>,
            Stride<Int<kSwizzledRowStride>, Int<kSwizzledColStride>>>{}));

    using CuteLayout = decltype(tile_to_shape(
        LayoutAtom{}, cute::Shape<Int<Layout_::kRows>, Int<Layout_::kCols>>{}));

    DEVICE Swizzled() : layout_(CuteLayout{}) {}

    DEVICE int operator()(int i, int j) const { return layout_(i, j); }

  private:
    CuteLayout layout_;
};

template <typename Layout>
static constexpr size_t num_rows = Layout::kRows;

template <typename Layout>
static constexpr size_t num_cols = Layout::kCols;

template <typename Layout>
static constexpr size_t row_stride = Layout::kRowStride;

template <typename Layout>
static constexpr size_t col_stride = Layout::kColStride;

template <typename Layout>
static constexpr size_t get_numel = Layout::kNumel;

// We wrap CuTe's `Layout`, which consists of `Shape` and `Stride`, into an
// intelligent row-major or column-major layout. In a row-major layout, the
// column stride is 1, whereas in a column-major layout, the row stride is 1.
// NOTE: A potential issue is that `ColMajor<1, 1>` will also be indentified as
// a row-major layout.
template <typename Layout_>
static constexpr Layout layout_type = Layout_::kType;

template <const bool kSwizzled, const int kShape1, const int kShape2,
          const int kStride1, const int kStride2, const int kB, const int kM,
          const int kS>
struct LayoutMaker {
    HOST_DEVICE auto operator()();
};

template <const int kShape1, const int kShape2, const int kStride1,
          const int kStride2, const int kB, const int kM, const int kS>
struct LayoutMaker<false, kShape1, kShape2, kStride1, kStride2, kB, kM, kS> {
    static_assert(kB == -1 && kS == -1 && kM == -1,
                  "Swizzling should not be set in matrix layout.");

    HOST_DEVICE auto operator()() {
        using Layout = MatrixLayout<kShape1, kShape2, kStride1, kStride2>;
        return Layout{};
    }
};

template <const int kShape1, const int kShape2, const int kStride1,
          const int kStride2, const int kB, const int kM, const int kS>
struct LayoutMaker<true, kShape1, kShape2, kStride1, kStride2, kB, kM, kS> {
    static_assert(kB >= 0 && kS >= 0 && kM >= 0,
                  "Swizzling should not be set in matrix layout.");

    HOST_DEVICE auto operator()() {
        using Layout = MatrixLayout<kShape1, kShape2, kStride1, kStride2>;
        return Swizzled<Layout, kB, kM, kS>{};
    }
};

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

}  // namespace tile_layout
}  // namespace tiledcuda::cell
