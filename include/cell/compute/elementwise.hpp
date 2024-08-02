#pragma once

#include "cuda_utils.hpp"
#include "types/layout.hpp"

#include <cute/algorithm/axpby.hpp>
#include <cute/config.hpp>
#include <cute/tensor.hpp>

namespace tiledcuda::cell::compute {

template <class XEngine, class XLayout>
DEVICE void cute_tanh(cute::Tensor<XEngine, XLayout>& tensor) {
#pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = tanh(tensor(i));
    }
}

template <class XEngine, class XLayout>
DEVICE void cute_sigmoid(cute::Tensor<XEngine, XLayout>& tensor) {
#pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = 1.0 / (1.0 + exp(-tensor(i)));
    }
}

template <typename RegTile, const tl::Layout kLayout>
struct ElementWise {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename Functor>
    DEVICE void operator()(RegTile& tile, Functor functor) {}
};

template <typename RegTile>
struct ElementWise<RegTile, tl::Layout::kRowMajor> {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename Functor>
    DEVICE void operator()(RegTile& tile, Functor functor) {
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                tile(i, j)(0, 0) = functor(tile(i, j)(0, 0));
                tile(i, j)(0, 1) = functor(tile(i, j)(0, 1));
                tile(i, j)(1, 0) = functor(tile(i, j)(1, 0));
                tile(i, j)(1, 1) = functor(tile(i, j)(1, 1));

                tile(i, j)(0, 2) = functor(tile(i, j)(0, 2));
                tile(i, j)(0, 3) = functor(tile(i, j)(0, 3));
                tile(i, j)(1, 2) = functor(tile(i, j)(1, 2));
                tile(i, j)(1, 3) = functor(tile(i, j)(1, 3));
            }
        }
    }
};

template <typename RegTile>
struct ElementWise<RegTile, tl::Layout::kColMajor> {
    using DType = typename RegTile::DType::DType;

    static constexpr int kRows = RegTile::kRows;
    static constexpr int kCols = RegTile::kCols;

    template <typename Functor>
    DEVICE void operator()(RegTile& tile, Functor functor) {}
};

template <typename RegTile, const tl::Layout kLayout>
struct Exp {
    using DType = typename RegTile::DType::DType;

    DEVICE void operator()(RegTile& tile) {
        ElementWise<RegTile, kLayout> element_wise;
        element_wise(tile, [](DType x) { return exp(x); });
    }
};

}  // namespace tiledcuda::cell::compute
