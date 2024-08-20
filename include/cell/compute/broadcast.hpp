#pragma once

#include "cell/compute/math_functor.hpp"
#include "cuda_utils.hpp"
#include "types/layout.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell::compute {
namespace tl = tile_layout;

template <typename SrcTile, typename DstTile, const tl::Layout kLayout>
struct Broadcast {
    using DType = typename DstTile::DType::DType;

    static constexpr int kRows = DstTile::kRows;
    static constexpr int kCols = DstTile::kCols;

    DEVICE void operator()(const SrcTile& src, DstTile& dst) {}
};

template <typename SrcTile, typename DstTile>
struct Broadcast<SrcTile, DstTile, tl::Layout::kRowMajor> {
    using DType = typename DstTile::DType::DType;

    static constexpr int kRows = DstTile::kRows;
    static constexpr int kCols = DstTile::kCols;

    DEVICE void operator()(const SrcTile& src, DstTile& dst) {
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
            DType top_row = src(i, 0);
            DType bottom_row = src(i, 1);
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                dst(i, j)(0, 0) = top_row;
                dst(i, j)(0, 1) = top_row;
                dst(i, j)(1, 0) = top_row;
                dst(i, j)(1, 1) = top_row;

                dst(i, j)(0, 2) = bottom_row;
                dst(i, j)(0, 3) = bottom_row;
                dst(i, j)(1, 2) = bottom_row;
                dst(i, j)(1, 3) = bottom_row;
            }
        }
    }
};

template <typename SrcTile, typename DstTile>
struct Broadcast<SrcTile, DstTile, tl::Layout::kColMajor> {
    using DType = typename DstTile::DType::DType;

    static constexpr int kRows = DstTile::kRows;
    static constexpr int kCols = DstTile::kCols;

    DEVICE void operator()(const SrcTile& src, DstTile& dst) {
#pragma unroll
        for (int j = 0; j < kCols; ++j) {
            DType top_col = src(0, j);
            DType bottom_col = src(1, j);
#pragma unroll
            for (int i = 0; i < kRows; ++i) {
                dst(i, j)(0, 0) = top_col;
                dst(i, j)(1, 0) = top_col;
                dst(i, j)(0, 1) = top_col;
                dst(i, j)(1, 1) = top_col;

                dst(i, j)(2, 0) = bottom_col;
                dst(i, j)(3, 0) = bottom_col;
                dst(i, j)(2, 1) = bottom_col;
                dst(i, j)(3, 1) = bottom_col;
            }
        }
    }
};

template <typename SrcTile, typename DstTile, typename Functor,
          const tl::Layout kLayout>
struct BroadcastFuse {
    using DType = typename DstTile::DType::DType;

    static constexpr int kRows = DstTile::kRows;
    static constexpr int kCols = DstTile::kCols;

    DEVICE void operator()(const SrcTile& src, DstTile& dst) {}
};

template <typename SrcTile, typename DstTile, typename Functor>
struct BroadcastFuse<SrcTile, DstTile, Functor, tl::Layout::kRowMajor> {
    using DType = typename DstTile::DType::DType;

    static constexpr int kRows = DstTile::kRows;
    static constexpr int kCols = DstTile::kCols;

    DEVICE void operator()(const SrcTile& src, DstTile& dst) {
        Functor f;
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
            DType top_row = src(i, 0);
            DType bottom_row = src(i, 1);
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                f(dst(i, j)(0, 0), top_row, dst(i, j)(0, 0));
                f(dst(i, j)(0, 1), top_row, dst(i, j)(0, 1));
                f(dst(i, j)(1, 0), top_row, dst(i, j)(1, 0));
                f(dst(i, j)(1, 1), top_row, dst(i, j)(1, 1));

                f(dst(i, j)(0, 2), bottom_row, dst(i, j)(0, 2));
                f(dst(i, j)(0, 3), bottom_row, dst(i, j)(0, 3));
                f(dst(i, j)(1, 2), bottom_row, dst(i, j)(1, 2));
                f(dst(i, j)(1, 3), bottom_row, dst(i, j)(1, 3));
            }
        }
    }
};

template <typename SrcTile, typename DstTile, typename Functor>
struct BroadcastFuse<SrcTile, DstTile, Functor, tl::Layout::kColMajor> {
    using DType = typename DstTile::DType::DType;

    static constexpr int kRows = DstTile::kRows;
    static constexpr int kCols = DstTile::kCols;

    DEVICE void operator()(const SrcTile& src, DstTile& dst) {
        Functor f;
#pragma unroll
        for (int j = 0; j < kCols; ++j) {
            DType top_col = src(0, j);
            DType bottom_col = src(1, j);
#pragma unroll
            for (int i = 0; i < kRows; ++i) {
                f(dst(i, j)(0, 0), top_col, dst(i, j)(0, 0));
                f(dst(i, j)(1, 0), top_col, dst(i, j)(1, 0));
                f(dst(i, j)(0, 1), top_col, dst(i, j)(0, 1));
                f(dst(i, j)(1, 1), top_col, dst(i, j)(1, 1));

                f(dst(i, j)(2, 0), bottom_col, dst(i, j)(2, 0));
                f(dst(i, j)(3, 0), bottom_col, dst(i, j)(3, 0));
                f(dst(i, j)(2, 1), bottom_col, dst(i, j)(2, 1));
                f(dst(i, j)(3, 1), bottom_col, dst(i, j)(3, 1));
            }
        }
    }
};

template <typename SrcTile, typename DstTile, const tl::Layout kLayout>
using BroadcastAdd =
    BroadcastFuse<SrcTile, DstTile, Add<typename SrcTile::DType>, kLayout>;

template <typename SrcTile, typename DstTile, const tl::Layout kLayout>
using BroadcastSub =
    BroadcastFuse<SrcTile, DstTile, Sub<typename SrcTile::DType>, kLayout>;

template <typename SrcTile, typename DstTile, const tl::Layout kLayout>
using BroadcastMul =
    BroadcastFuse<SrcTile, DstTile, Mul<typename SrcTile::DType>, kLayout>;

template <typename SrcTile, typename DstTile, const tl::Layout kLayout>
using BroadcastDiv =
    BroadcastFuse<SrcTile, DstTile, Div<typename SrcTile::DType>, kLayout>;

}  // namespace tiledcuda::cell::compute
