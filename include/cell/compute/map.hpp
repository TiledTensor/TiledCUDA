#pragma once

#include "cell/compute/math_functor.hpp"
#include "cell/warp.hpp"
#include "cuda_utils.hpp"
#include "types/layout.hpp"

namespace tiledcuda::cell::compute {

namespace tl = tile_layout;

namespace detail {

template <typename RegTile, const tl::Layout kLayout>
struct UnaryMap {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename Functor>
    DEVICE void operator()(const RegTile& src, RegTile& dst, Functor functor) {}
};

template <typename RegTile>
struct UnaryMap<RegTile, tl::Layout::kRowMajor> {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename Functor>
    DEVICE void operator()(const RegTile& src, RegTile& dst, Functor functor) {
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                dst(i, j)(0, 0) = functor(src(i, j)(0, 0));
                dst(i, j)(0, 1) = functor(src(i, j)(0, 1));
                dst(i, j)(1, 0) = functor(src(i, j)(1, 0));
                dst(i, j)(1, 1) = functor(src(i, j)(1, 1));
                dst(i, j)(0, 2) = functor(src(i, j)(0, 2));
                dst(i, j)(0, 3) = functor(src(i, j)(0, 3));
                dst(i, j)(1, 2) = functor(src(i, j)(1, 2));
                dst(i, j)(1, 3) = functor(src(i, j)(1, 3));
            }
        }
    }
};

template <typename RegTile>
struct UnaryMap<RegTile, tl::Layout::kColMajor> {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename Functor>
    DEVICE void operator()(const RegTile& src, RegTile& dst, Functor functor) {
#pragma unroll
        for (int j = 0; j < kCols; ++j) {
#pragma unroll
            for (int i = 0; i < kRows; ++i) {
                dst(i, j)(0, 0) = functor(src(i, j)(0, 0));
                dst(i, j)(1, 0) = functor(src(i, j)(1, 0));
                dst(i, j)(0, 1) = functor(src(i, j)(0, 1));
                dst(i, j)(1, 1) = functor(src(i, j)(1, 1));
                dst(i, j)(0, 2) = functor(src(i, j)(2, 0));
                dst(i, j)(1, 2) = functor(src(i, j)(3, 0));
                dst(i, j)(0, 3) = functor(src(i, j)(2, 1));
                dst(i, j)(1, 3) = functor(src(i, j)(3, 1));
            }
        }
    }
};

template <typename RegTile, const tl::Layout kLayout>
struct BinaryMap {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename Functor>
    DEVICE void operator()(const RegTile& lhs, const RegTile& rhs, RegTile& dst,
                           Functor functor) {}
};

template <typename RegTile>
struct BinaryMap<RegTile, tl::Layout::kRowMajor> {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename Functor>
    DEVICE void operator()(const RegTile& lhs, const RegTile& rhs, RegTile& dst,
                           Functor functor) {
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                dst(i, j)(0, 0) = functor(lhs(i, j)(0, 0), rhs(i, j)(0, 0));
                dst(i, j)(0, 1) = functor(lhs(i, j)(0, 1), rhs(i, j)(0, 1));
                dst(i, j)(1, 0) = functor(lhs(i, j)(1, 0), rhs(i, j)(1, 0));
                dst(i, j)(1, 1) = functor(lhs(i, j)(1, 1), rhs(i, j)(1, 1));
                dst(i, j)(0, 2) = functor(lhs(i, j)(0, 2), rhs(i, j)(0, 2));
                dst(i, j)(0, 3) = functor(lhs(i, j)(0, 3), rhs(i, j)(0, 3));
                dst(i, j)(1, 2) = functor(lhs(i, j)(1, 2), rhs(i, j)(1, 2));
                dst(i, j)(1, 3) = functor(lhs(i, j)(1, 3), rhs(i, j)(1, 3));
            }
        }
    }
};

template <typename RegTile>
struct BinaryMap<RegTile, tl::Layout::kColMajor> {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename Functor>
    DEVICE void operator()(const RegTile& lhs, const RegTile& rhs, RegTile& dst,
                           Functor functor) {
#pragma unroll
        for (int j = 0; j < kCols; ++j) {
#pragma unroll
            for (int i = 0; i < kRows; ++i) {
                dst(i, j)(0, 0) = functor(lhs(i, j)(0, 0), rhs(i, j)(0, 0));
                dst(i, j)(1, 0) = functor(lhs(i, j)(1, 0), rhs(i, j)(1, 0));
                dst(i, j)(0, 1) = functor(lhs(i, j)(0, 1), rhs(i, j)(0, 1));
                dst(i, j)(1, 1) = functor(lhs(i, j)(1, 1), rhs(i, j)(1, 1));
                dst(i, j)(2, 0) = functor(lhs(i, j)(2, 0), rhs(i, j)(2, 0));
                dst(i, j)(2, 1) = functor(lhs(i, j)(3, 0), rhs(i, j)(3, 0));
                dst(i, j)(3, 0) = functor(lhs(i, j)(2, 1), rhs(i, j)(2, 1));
                dst(i, j)(3, 1) = functor(lhs(i, j)(3, 1), rhs(i, j)(3, 1));
            }
        }
    }
};

template <typename RegTile, const tl::Layout kLayout>
struct ReduceMap {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename ReduceTile, typename Functor>
    DEVICE void operator()(const RegTile& src, ReduceTile& reduce_tile,
                           RegTile& dst, Functor functor) {}
};

template <typename RegTile>
struct ReduceMap<RegTile, tl::Layout::kRowMajor> {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename ReduceTile, typename Functor>
    DEVICE void operator()(const RegTile& src, ReduceTile& reduce_tile,
                           RegTile& dst, Functor functor) {
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
            DType top_row_sum = reduce_tile(i, 0);
            DType bottom_row_sum = reduce_tile(i, 1);
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                tile(i, j)(0, 0) = functor(src(i, j)(0, 0), top_row_sum);
                tile(i, j)(0, 1) = functor(src(i, j)(0, 1), top_row_sum);
                tile(i, j)(1, 0) = functor(src(i, j)(1, 0), top_row_sum);
                tile(i, j)(1, 1) = functor(src(i, j)(1, 1), top_row_sum);

                tile(i, j)(0, 2) = functor(src(i, j)(0, 2), bottom_row_sum);
                tile(i, j)(0, 3) = functor(src(i, j)(0, 3), bottom_row_sum);
                tile(i, j)(1, 2) = functor(src(i, j)(1, 2), bottom_row_sum);
                tile(i, j)(1, 3) = functor(src(i, j)(1, 3), bottom_row_sum);
            }
        }
    }
};

}  // namespace detail

template <typename RegTile, const tl::Layout kLayout>
struct RegExp {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    DEVICE void operator()(const RegTile& src, RegTile& dst) {
        detail::UnaryMap<RegTile, kLayout> row_exp;
        row_exp(src, dst, Exp<DType>{});
    }
};

template <typename RegTile, const tl::Layout kLayout>
struct RegLog {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    DEVICE void operator()(const RegTile& src, RegTile& dst) {
        detail::UnaryMap<RegTile, kLayout> row_log;
        row_log(src, dst, Log<DType>{});
    }
};
}  // namespace detail

template <typename RegTile, const tl::Layout kLayout>
struct RegSub {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    DEVICE void operator()(const RegTile& lhs, const RegTile& rhs,
                           RegTile& dst) {
        detail::BinaryMap<RegTile, kLayout> row_sub;
        row_sub(lhs, rhs, dst, Sub<DType>{});
    }
};

template <typename RegTile, const tl::Layout kLayout>
struct RegSubReduce {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename ReduceTile>
    DEVICE void operator()(const RegTile& src, ReduceTile& reduce_tile,
                           RegTile& dst) {
        detail::ReduceMap<RegTile, kLayout> row_sub_reduce_max;
        row_sub_reduce_max(src, reduce_tile, dst, Sub<DType>{});
    }
};
}

}  // namespace tiledcuda::cell::compute