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

    static constexpr Layout layout_type =
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

/// @brief Swizzled layout for 16x16 BaseTile.
template <const int kElemBits, typename AtomLayout>
struct SwizzledRowMajor;

// @brief: leverage CuTe's swizzle functions, swizzle(B, M, S), permute elements
//         in a 2D coordinate space. This 2D coordinate space has 2^B rows and
//         2^S columns, and each coordinate position has 2^M elements.
//         Therefore, to apply a swizzle function to a 2D data tile, the data
//         tile should have a shape that is a multiple of 2^B x 2^S x 2^M.
template <typename AtomLayout>
struct SwizzledRowMajor<16, AtomLayout> {
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

    DEVICE SwizzledRowMajor()
        : swizzled_(SwizzledBaseTile{}), layout_(AtomLayout{}){};

    DEVICE int operator()(int i, int j) const {
        int s = swizzled_(i, j);
        return layout_(s / BaseShape::kCols, s % BaseShape::kCols);
    }

  private:
    SwizzledBaseTile swizzled_;
    AtomLayout layout_;
};

/// @brief Swizzled column-major layout for 16x16 BaseTile.
template <const int kElemBits, typename AtomLayout>
struct SwizzledColMajor;

template <typename AtomLayout>
struct SwizzledColMajor<16, AtomLayout> {
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

    DEVICE SwizzledColMajor()
        : swizzled_(SwizzledBaseTile{}), layout_(AtomLayout{}){};

    DEVICE int operator()(int i, int j) const {
        int s = swizzled_(i, j);
        return layout_(s % BaseShape::kRows, s / BaseShape::kRows);
    }

  private:
    SwizzledBaseTile swizzled_;
    AtomLayout layout_;
};

template <typename Shared, const bool kSwizzled, const Layout kType,
          const int kSizeOfTypeBits>
struct SharedLayoutWrapperImpl;

/// @brief Shared memory layout for non-swizzled layout with 32-bit data type.
template <typename Shared, const Layout kType, const int kSizeOfTypeBits>
struct SharedLayoutWrapperImpl<Shared, false, kType, kSizeOfTypeBits> {
    using BaseShape = traits::BaseTileShape<typename Shared::DType>;
    using Layout =
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<Int<Shared::kRowStride>, Int<Shared::kColStride>>>;
};

/// @brief Shared memory layout for swizzled row-major layout with 16-bit data
///        type.
template <typename Shared>
struct SharedLayoutWrapperImpl<Shared, true, Layout::kRowMajor, 16> {
    using BaseShape = traits::BaseTileShape<typename Shared::DType>;
    using LayoutAtom =
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<Int<Shared::kRowStride>, Int<Shared::kColStride>>>;

    using Layout = SwizzledRowMajor<16, LayoutAtom>;
};

/// @brief Shared memory layout for swizzled col-major layout with 16-bit data
///        type.
template <typename Shared>
struct SharedLayoutWrapperImpl<Shared, true, Layout::kColMajor, 16> {
    using BaseShape = traits::BaseTileShape<typename Shared::DType>;
    using LayoutAtom =
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<Int<Shared::kRowStride>, Int<Shared::kColStride>>>;

    using Layout = SwizzledColMajor<16, LayoutAtom>;
};

///// ======= shared memory layout for storer ========

/// @brief Swizzled layout for 16x16 BaseTile.
template <const int kElemBits, typename AtomLayout>
struct SwizzledRowMajorStorer;

template <typename AtomLayout>
struct SwizzledRowMajorStorer<32, AtomLayout> {
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

    DEVICE SwizzledRowMajorStorer()
        : swizzled_(SwizzledBaseTile{}), layout_(AtomLayout{}){};

    DEVICE int operator()(int i, int j) const {
        int s = swizzled_(i, j);
        return layout_(s / BaseShape::kCols, s % BaseShape::kCols);
    }

  private:
    SwizzledBaseTile swizzled_;
    AtomLayout layout_;
};

/// @brief Swizzled layout for 16x16 BaseTile.
template <const int kElemBits, typename AtomLayout>
struct SwizzledColMajorStorer;

template <typename AtomLayout>
struct SwizzledColMajorStorer<32, AtomLayout> {
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

    DEVICE SwizzledColMajorStorer()
        : swizzled_(SwizzledBaseTile{}), layout_(AtomLayout{}){};

    DEVICE int operator()(int i, int j) const {
        int s = swizzled_(i, j);
        return layout_(s % BaseShape::kRows, s / BaseShape::kRows);
    }

  private:
    SwizzledBaseTile swizzled_;
    AtomLayout layout_;
};

template <typename Shared, const bool kSwizzled, const Layout kType,
          const int kSizeOfTypeBits>
struct SharedStorerLayoutWrapperImpl;

/// @brief Shared memory layout for non-swizzled layout with 16-bit data type.
template <typename Shared, const Layout kType, const int kSizeOfTypeBits>
struct SharedStorerLayoutWrapperImpl<Shared, false, kType, kSizeOfTypeBits> {
    using Layout =
        cute::Layout<Shape<Int<Shared::kRows>, Int<Shared::kCols>>,
                     Stride<Int<Shared::kRowStride>, Int<Shared::kColStride>>>;
};

/// @brief Shared memory layout for swizzled layout with 32-bit data type.
template <typename Shared>
struct SharedStorerLayoutWrapperImpl<Shared, true, Layout::kRowMajor, 16> {
    using LayoutAtom =
        cute::Layout<Shape<Int<Shared::kRows>, Int<Shared::kCols>>,
                     Stride<Int<Shared::kRowStride>, Int<Shared::kColStride>>>;

    using Layout = SwizzledRowMajor<16, LayoutAtom>;
};

/// @brief Shared memory layout for swizzled col-major layout with 16-bit data
///        type.
template <typename Shared>
struct SharedStorerLayoutWrapperImpl<Shared, true, Layout::kColMajor, 16> {
    using LayoutAtom =
        cute::Layout<Shape<Int<Shared::kRows>, Int<Shared::kCols>>,
                     Stride<Int<Shared::kRowStride>, Int<Shared::kColStride>>>;

    using Layout = SwizzledColMajor<16, LayoutAtom>;
};

/// @brief Shared memory layout for swizzled layout with 32-bit data type.
template <typename Shared>
struct SharedStorerLayoutWrapperImpl<Shared, true, Layout::kRowMajor, 32> {
    using BaseShape = traits::BaseTileShape<typename Shared::DType>;
    using LayoutAtom =
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<Int<Shared::kRowStride>, Int<Shared::kColStride>>>;

    using Layout = SwizzledRowMajorStorer<32, LayoutAtom>;
};

/// @brief Shared memory layout for swizzled col-major layout with 16-bit data
///        type.
template <typename Shared>
struct SharedStorerLayoutWrapperImpl<Shared, true, Layout::kColMajor, 32> {
    using BaseShape = traits::BaseTileShape<typename Shared::DType>;
    using LayoutAtom =
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<Int<Shared::kRowStride>, Int<Shared::kColStride>>>;

    using Layout = SwizzledColMajorStorer<32, LayoutAtom>;
};
}  // namespace detail

/// @brief: Wapper for creating non-swizzled or swizzled shared memory layout.
template <typename Shared>
struct SharedLayoutWrapper {
    static constexpr int kSizeOfTypeBits = int(sizeof(Shared::DType) * 8);

    using Layout =
        detail::SharedLayoutWrapperImpl<Shared, Shared::kSwizzled,
                                        Shared::kType, kSizeOfTypeBits>::Layout;
};

/// FIXME: This is a temporary solution for storing shared memory with bank
/// conflict free. Make the shared memory layout consistent with the swizzled
/// layout.
template <typename Shared>
struct SharedStorerLayoutWrapper {
    static constexpr int kSizeOfTypeBits = int(sizeof(Shared::DType) * 8);

    using Layout = detail::SharedStorerLayoutWrapperImpl<
        Shared, Shared::kSwizzled, Shared::kType, kSizeOfTypeBits>::Layout;
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
static constexpr Layout layout_type = Layout_::layout_type;

template <const int kShape1, const int kShape2, const int kStride1,
          const int kStride2>
HOST_DEVICE auto make_tile_layout() {
    using Layout = MatrixLayout<kShape1, kShape2, kStride1, kStride2>;
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
