#pragma once

#include "types/tile_shape.hpp"

namespace tiledcuda::cell {

namespace tl = tile_layout;

struct Underscore {};                  // dummy type for underscore
static const __device__ Underscore _;  // for slicing

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

    // Since a Tile is considered to be at most a 2D array, the iterator
    // traverses over these two dimensions. The current rules are:
    // 1. If the index is a 2D integer, this access is considered to be a
    //    single tile, hence it returns a Tile.
    // 2. If any part of the index is an underscore, this access is
    //    considered to be a slice, naturally it returns a TileIterator.

    DEVICE Tile* operator()(int x, int y) { return nullptr; }

    // FIXME(haruhi): this is a placeholder to bypass compiling. The return
    // types here needs to be computed based on the input arguments.
    DEVICE auto operator()(int x, const Underscore& y) {
        TileIterator<Tile, ChunkShape> iter(data_);
        return iter;
    }

    // FIXME(haruhi): this is a placeholder to bypass compiling. The return
    // types here needs to be computed based on the input arguments.
    DEVICE auto operator()(const Underscore& x, int y) {
        TileIterator<Tile, ChunkShape> iter(data_);
        return iter;
    }

    // FIXME(haruhi): this is a placeholder to bypass compiling. The return
    // types here needs to be computed based on the input arguments.
    DEVICE auto to_tile() {
        Tile tile(data_);
        return tile;
    }

  private:
    typename Tile::DType* data_;
};

}  // namespace tiledcuda::cell
