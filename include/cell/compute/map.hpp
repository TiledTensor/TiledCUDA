#pragma once

#include "cell/compute/math_functor.hpp"
#include "cell/warp.hpp"
#include "cuda_utils.hpp"
#include "types/layout.hpp"

namespace tiledcuda::cell::compute {

namespace tl = tile_layout;

namespace detail {

template <typename RegTile, typename Functor>
struct ElementWise {};

}  // namespace detail
}  // namespace tiledcuda::cell::compute