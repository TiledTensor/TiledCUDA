#pragma once

#include "types/tile_shape.hpp"

namespace tiledcuda::cell {

namespace tl = tile_layout;

struct Underscore {};                  // dummy type for underscore
static const __device__ Underscore _;  // for slicing

template <class Tile, class ChunkShape>
struct TileIterator;

/// @brief TileIterator is to chunk a tile into smaller tiles, and iterate over
///        the smaller sub-tiles.
/// @tparam Tile_ the tile type.
/// @tparam ChunkShape_ the shape of the chunk.
template <class Tile_, class ChunkShape_>
struct TileIterator {
    using Tile = Tile_;
    using ChunkShape = ChunkShape_;

    static_assert(Tile::kRows >= dim_size<0, ChunkShape>,
                  "Tile::kRows must be >= dim_size<0, ChunkShape>");
    static_assert(Tile::kCols >= dim_size<1, ChunkShape>,
                  "Tile::kCols must be >= dim_size<1, ChunkShape>");
    static constexpr int sc0 = Tile::kRows / dim_size<0, ChunkShape>;
    static constexpr int sc1 = Tile::kCols / dim_size<1, ChunkShape>;

    DEVICE TileIterator(typename Tile::DType* data) : data_(data) {}

    DEVICE Tile* operator()(int x, int y) { return nullptr; }

    DEVICE TileIterator* operator()(int x, const Underscore& y) {
        return nullptr;
    }

    DEVICE TileIterator* operator()(const Underscore& x, int y) {
        return nullptr;
    }

  private:
    typename Tile::DType* data_;
};

}  // namespace tiledcuda::cell
