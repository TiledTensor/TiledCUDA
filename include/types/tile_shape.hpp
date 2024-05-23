#pragma once

#include <cute/container/tuple.hpp>
#include <cute/int_tuple.hpp>

namespace tiledcuda::cell {

template <size_t... T>
using TileShape = cute::tuple<std::integral_constant<size_t, T>...>;

template <const size_t I, typename TileShape>
inline static constexpr size_t dim_size = cute::get<I>(TileShape{});

template <typename TileShape>
inline static constexpr int64_t get_numel = cute::size(TileShape{});

}  // namespace tiledcuda::cell
