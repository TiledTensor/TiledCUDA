#pragma once
#include "cell/traits/base.hpp"
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

namespace detail {
using namespace cute;

template <const int kRows_, const int kCols_, const int kRowStride_,
          const int kColStride_, const Layout kType_>
struct SharedLayout {
    using BaseShape = traits::BaseTileShape<float>;

    static constexpr int kRows = kRows_;
    static constexpr int kCols = kCols_;

    static constexpr int kRowStride = kRowStride_;
    static constexpr int kColStride = kColStride_;

    static constexpr int kNumel = kRows * kCols;

    static constexpr Layout kType = kType_;

    DEVICE int operator()(int i, int j) const {
        int tile_x = i / BaseShape::kRows;
        int tile_y = j / BaseShape::kCols;

        int in_tile_x = i % BaseShape::kRows;
        int in_tile_y = j % BaseShape::kCols;

        int tile_offset = tile_x * kRowStride + tile_y * kColStride;
        int in_tile_offset = in_tile_(in_tile_x, in_tile_y);

        return tile_offset + in_tile_offset;
    }

  private:
    using BaseTileLayout = std::conditional_t<
        kType == Layout::kRowMajor,
        cute::Layout<Shape<_16, _16>, Stride<_16, _1>>,  /*RowMajor*/
        cute::Layout<Shape<_16, _16>, Stride<_1, _16>>>; /*ColMajor*/
    BaseTileLayout in_tile_;
};

/// @brief Swizzled layout for 16x16 BaseTile.
template <const int kBitsPerAccess>
struct SwizzledRowMajor;

template <>
struct SwizzledRowMajor<64> {
    using BaseShape = traits::BaseTileShape<float>;

    static constexpr int kB = 2;
    static constexpr int kM = 2;
    static constexpr int kS = 3;

    using SwizzledAtom =
        decltype(composition(cute::Swizzle<kB, kM, kS>{},
                             cute::Layout<Shape<_16, _8>, Stride<_8, _1>>{}));
    using SwizzledBaseTile = decltype(tile_to_shape(
        SwizzledAtom{},
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<Int<BaseShape::kCols>, _1>>{}));

    DEVICE SwizzledRowMajor() : swizzled_(SwizzledBaseTile{}){};

    DEVICE int operator()(int i, int j) const { return swizzled_(i, j); }

  private:
    SwizzledBaseTile swizzled_;
};

// @brief: leverage CuTe's swizzle functions, swizzle(B, M, S), permute elements
//         in a 2D coordinate space. This 2D coordinate space has 2^B rows and
//         2^S columns, and each coordinate position has 2^M elements.
//         Therefore, to apply a swizzle function to a 2D data tile, the data
//         tile should have a shape that is a multiple of 2^B x 2^S x 2^M.
template <>
struct SwizzledRowMajor<128> {
    using BaseShape = traits::BaseTileShape<__half>;

    static constexpr int kB = 2;
    static constexpr int kM = 3;
    static constexpr int kS = 3;

    static_assert(
        BaseShape::kNumel == ((1 << kB) * (1 << kM) * (1 << kS)),
        "Swizzling is performed based on the BaseTile, and the number of "
        "elements in a BaseTile should be equal to 2^B x 2^S x 2^M.");

    using SwizzledBaseTile = decltype(composition(
        cute::Swizzle<kB, kM, kS>{},
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<Int<BaseShape::kCols>, _1>>{}));

    DEVICE SwizzledRowMajor() : swizzled_(SwizzledBaseTile{}){};

    DEVICE int operator()(int i, int j) const { return swizzled_(i, j); }

  private:
    SwizzledBaseTile swizzled_;
};

/// @brief Swizzled column-major layout for 16x16 BaseTile.
template <const int kBitsPerAccess>
struct SwizzledColMajor;

template <>
struct SwizzledColMajor<64> {
    using BaseShape = traits::BaseTileShape<__half>;

    static constexpr int kB = 2;
    static constexpr int kM = 2;
    static constexpr int kS = 3;

    using SwizzledAtom =
        decltype(composition(cute::Swizzle<kB, kM, kS>{},
                             cute::Layout<Shape<_16, _8>, Stride<_1, _16>>{}));

    using SwizzledBaseTile = decltype(tile_to_shape(
        SwizzledAtom{},
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<_1, Int<BaseShape::kRows>>>{}));

    DEVICE SwizzledColMajor() : swizzled_(SwizzledBaseTile{}){};

    DEVICE int operator()(int i, int j) const { return swizzled_(i, j); }

  private:
    SwizzledBaseTile swizzled_;
};

template <>
struct SwizzledColMajor<128> {
    using BaseShape = traits::BaseTileShape<__half>;

    static constexpr int kB = 2;
    static constexpr int kM = 3;
    static constexpr int kS = 3;

    static_assert(
        BaseShape::kNumel == ((1 << kB) * (1 << kM) * (1 << kS)),
        "Swizzling is performed based on the BaseTile, and the number of "
        "elements in a BaseTile should be equal to 2^B x 2^S x 2^M.");

    using SwizzledBaseTile = decltype(composition(
        cute::Swizzle<kB, kM, kS>{},
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<_1, Int<BaseShape::kRows>>>{}));

    DEVICE SwizzledColMajor() : swizzled_(SwizzledBaseTile{}){};

    DEVICE int operator()(int i, int j) const { return swizzled_(i, j); }

  private:
    SwizzledBaseTile swizzled_;
};

template <const bool kSwizzled, const Layout kType, const int kBitsPerAcces>
struct SharedLayoutWrapperImpl;

/// @brief Shared memory layout for non-swizzled layout with 32-bit data type.
template <const int kBitsPerAcces>
struct SharedLayoutWrapperImpl<false, Layout::kRowMajor, kBitsPerAcces> {
    using BaseShape = traits::BaseTileShape<__half>;

    using Layout =
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<Int<BaseShape::kCols>, _1>>;
};

/// @brief Shared memory layout for non-swizzled layout with 32-bit data type.
template <const int kBitsPerAcces>
struct SharedLayoutWrapperImpl<false, Layout::kColMajor, kBitsPerAcces> {
    using BaseShape = traits::BaseTileShape<__half>;
    using Layout =
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<_1, Int<BaseShape::kRows>>>;
};

/// @brief Shared memory layout for swizzled row-major layout with 16-bit data
///        type.
template <>
struct SharedLayoutWrapperImpl<true, Layout::kRowMajor, 64> {
    using BaseShape = traits::BaseTileShape<__half>;

    using Layout = SwizzledRowMajor<64>;
};

/// @brief Shared memory layout for swizzled col-major layout with 16-bit data
///        type.
template <>
struct SharedLayoutWrapperImpl<true, Layout::kColMajor, 64> {
    using BaseShape = traits::BaseTileShape<__half>;

    using Layout = SwizzledColMajor<64>;
};

/// @brief Shared memory layout for swizzled row-major layout with 16-bit data
///        type.
template <>
struct SharedLayoutWrapperImpl<true, Layout::kRowMajor, 128> {
    using BaseShape = traits::BaseTileShape<__half>;

    using Layout = SwizzledRowMajor<128>;
};

/// @brief Shared memory layout for swizzled col-major layout with 16-bit data
///        type.
template <>
struct SharedLayoutWrapperImpl<true, Layout::kColMajor, 128> {
    using BaseShape = traits::BaseTileShape<__half>;

    using Layout = SwizzledColMajor<128>;
};
}  // namespace detail

/// @brief: Wapper for creating non-swizzled or swizzled shared memory layout.
template <typename Shared, const int kBitsPerAccess>
struct SharedLayoutWrapper {
    using Layout =
        detail::SharedLayoutWrapperImpl<Shared::kSwizzled, Shared::kType,
                                        kBitsPerAccess>::Layout;
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

template <const int kShape1, const int kShape2, const int kStride1,
          const int kStride2>
HOST_DEVICE auto make_tile_layout() {
    using Layout = MatrixLayout<kShape1, kShape2, kStride1, kStride2>;
    return Layout{};
}

template <const int kShape1, const int kShape2, const int kStride1,
          const int kStride2, const Layout kType>
HOST_DEVICE auto make_shared_tile_layout() {
    using Layout =
        detail::SharedLayout<kShape1, kShape2, kStride1, kStride2, kType>;

    return Layout{};
}

HOST_DEVICE auto make_row_major_layout(const int row, const int col,
                                       const int stride) {
    return cute::make_layout(cute::make_shape(row, col),
                             cute::make_stride(stride, cute::_1{}));
}

HOST_DEVICE auto make_col_major_layout(const int row, const int col,
                                       const int stride) {
    return cute::make_layout(cute::make_shape(row, col),
                             cute::make_stride(cute::_1{}, stride));
}

}  // namespace tile_layout
}  // namespace tiledcuda::cell
