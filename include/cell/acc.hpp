#pragma once
#include "cuda_utils.hpp"

#include <cute/layout.hpp>

namespace tiledcuda::cell {
using namespace cute;

template <const int kM, const int kN, typename TiledMma>
DEVICE auto get_acc(const TiledMma& tiled_mma) {
    auto acc = partition_fragment_C(tiled_mma, Shape<Int<kM>, Int<kN>>{});
    clear(acc);

    return acc;
}
}  // namespace tiledcuda::cell
