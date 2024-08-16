#pragma once

#include "cell/compute/math_functor.hpp"
#include "cell/warp.hpp"
#include "cuda_utils.hpp"
#include "types/layout.hpp"

namespace tiledcuda::cell::compute {

namespace tl = tile_layout;

namespace detail {

// TODO(KuangjuX): Distinguish whether the `Layout` is Row Major or Column
// Major. Different Layouts have different directions of memory continuity,
// which will affect the memory access performance.
template <typename RegTile, typename Functor>
struct ElementWise {
    using DType = typename RegTile::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    DEVICE void operator()(const RegTile& src, RegTile& dst) {
        Functor f;
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                f(src(i, j), dst(i, j));
            }
        }
    }
};

// TODO(KuangjuX): Distinguish whether the `Layout` is Row Major or Column
// Major. Different Layouts have different directions of memory continuity,
// which will affect the memory access performance.
template <typename SrcRegTile, typename DstRegTile, typename Functor>
struct ElementWise2 {
    static constexpr int kRows = SrcRegTile::kRows;
    static constexpr int kCols = SrcRegTile::kCols;

    static_assert(kRows == DstRegTile::kRows, "kRows must be equal");
    static_assert(kCols == DstRegTile::kCols, "kCols must be equal");

    DEVICE void operator()(const SrcRegTile& src, DstRegTile& dst) {
        Functor f;
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                f(src(i, j), dst(i, j));
            }
        }
    }
};

// TODO(KuangjuX): Distinguish whether the `Layout` is Row Major or Column
// Major. Different Layouts have different directions of memory continuity,
// which will affect the memory access performance.
template <typename RegTile, typename Functor>
struct Binary {
    using DType = typename RegTile::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    DEVICE void operator()(const RegTile& lhs, const RegTile& rhs,
                           RegTile& dst) {
        Functor f;
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                f(lhs(i, j), rhs(i, j), dst(i, j));
            }
        }
    }
};

}  // namespace detail

template <typename RegTile>
using BaseTileExp = detail::ElementWise<RegTile, Exp<typename RegTile::DType>>;
template <typename RegTile>
using RegTileExp =
    detail::ElementWise<RegTile, BaseTileExp<typename RegTile::DType>>;

template <typename RegTile>
using BaseTileLog = detail::ElementWise<RegTile, Log<typename RegTile::DType>>;
template <typename RegTile>
using RegTileLog =
    detail::ElementWise<RegTile, BaseTileLog<typename RegTile::DType>>;

template <typename SrcRegTile, typename DstRegTile>
using BaseTileConvert = detail::ElementWise2<
    SrcRegTile, DstRegTile,
    Convert<typename SrcRegTile::DType, typename DstRegTile::DType>>;
template <typename SrcRegTile, typename DstRegTile>
using RegTileConvert = detail::ElementWise2<
    SrcRegTile, DstRegTile,
    BaseTileConvert<typename SrcRegTile::DType, typename DstRegTile::DType>>;

template <typename RegTile>
using BaseTileAdd = detail::Binary<RegTile, Add<typename RegTile::DType>>;
template <typename RegTile>
using RegTileAdd =
    detail::Binary<RegTile, BaseTileAdd<typename RegTile::DType>>;

template <typename RegTile>
using BaseTileSub = detail::Binary<RegTile, Sub<typename RegTile::DType>>;
template <typename RegTile>
using RegTileSub =
    detail::Binary<RegTile, BaseTileSub<typename RegTile::DType>>;

template <typename RegTile>
using BaseTileMul = detail::Binary<RegTile, Mul<typename RegTile::DType>>;
template <typename RegTile>
using RegTileMul =
    detail::Binary<RegTile, BaseTileMul<typename RegTile::DType>>;

template <typename RegTile>
using BaseTileDiv = detail::Binary<RegTile, Div<typename RegTile::DType>>;
template <typename RegTile>
using RegTileDiv =
    detail::Binary<RegTile, BaseTileDiv<typename RegTile::DType>>;

template <typename RegTile>
using BaseTileMax = detail::Binary<RegTile, Max<typename RegTile::DType>>;
template <typename RegTile>
using RegTileMax =
    detail::Binary<RegTile, BaseTileMax<typename RegTile::DType>>;

}  // namespace tiledcuda::cell::compute
