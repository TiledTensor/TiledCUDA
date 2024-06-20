#pragma once

#include "types/tile_shape.hpp"

namespace tiledcuda::cell {

namespace tl = tile_layout;

struct Underscore {};                  // dummy type for underscore
static const __device__ Underscore _;  // for slicing

template <class Tile, class ChunkShape>
struct SharedTileIterator;

/// @brief TileIterator is to chunk a tile into smaller tiles, and iterate over
///        the smaller sub-tiles.
/// @tparam Tile_ the tile type.
/// @tparam ChunkShape_ the shape of the chunk.
template <class Tile_, class ChunkShape_>
class SharedTileIterator {
  public:
    using Tile = Tile_;
    using ChunkShape = ChunkShape_;

    static_assert(Tile::kRows >= dim_size<0, ChunkShape>,
                  "Tile::kRows must be >= dim_size<0, ChunkShape>");
    static_assert(Tile::kCols >= dim_size<1, ChunkShape>,
                  "Tile::kCols must be >= dim_size<1, ChunkShape>");

    static constexpr int sc0 = Tile::kRows / dim_size<0, ChunkShape>;
    static constexpr int sc1 = Tile::kCols / dim_size<1, ChunkShape>;

    // constants for construct the new `Tile` for slicing
    static constexpr int row_stride = tl::row_stride<typename Tile::Layout>;
    static constexpr int col_stride = tl::col_stride<typename Tile::Layout>;

    static constexpr bool kIsRowMajor = col_stride == 1;

    DEVICE SharedTileIterator(typename Tile::DType* data) : data_(data) {}

    // Since a Tile is considered to be at most a 2D array, the iterator
    // traverses over these two dimensions. The current rules are:
    // 1. If the index is a 2D integer, this access is considered to be a
    //    single tile, hence it returns a Tile.
    // 2. If any part of the index is an underscore, this access is
    //    considered to be a slice, naturally it returns a TileIterator.

    DEVICE auto operator()(int x) {
        // FIXME(haruhi): placeholder implementation.
        static_assert(sc0 == 1 || sc1 == 1,
                      "A single index is supported only when the stripe count "
                      "of one of the iterator's dimensions is 1.");
        Tile tile(data_);

        return tile;
    }

    DEVICE auto operator()(int x, int y) {
        // placeholder
        Tile tile(data_);

        return tile;
    }

    // FIXME(haruhi): this is a placeholder to bypass compiling. The return
    // types here needs to be computed based on the input arguments.
    DEVICE auto operator()(int x, const Underscore& y) {
        const int row = dim_size<0, ChunkShape>;
        const int col = Tile::kCols;

        using NewLayout =
            decltype(tl::make_tile_layout<row, col, row_stride, col_stride>());

        using NewTile = SharedTile<typename Tile::DType, NewLayout>;
        using Iter = SharedTileIterator<NewTile, ChunkShape>;
        static_assert(Iter::sc0 == 1);

        int offset = kIsRowMajor ? x * Tile::kCols * dim_size<1, ChunkShape>
                                 : x * row_stride;

        Iter iter(data_ + offset);
        return iter;
    }

    // FIXME(haruhi): this is a placeholder to bypass compiling. The return
    // types here needs to be computed based on the input arguments.
    DEVICE auto operator()(const Underscore& x, int y) {
        const int row = Tile::kRows;
        const int col = dim_size<1, ChunkShape>;

        using NewLayout =
            decltype(tl::make_tile_layout<row, col, row_stride, col_stride>());

        using NewTile = SharedTile<typename Tile::DType, NewLayout>;
        using Iter = SharedTileIterator<NewTile, ChunkShape>;
        static_assert(Iter::sc1 == 1);

        int offset = kIsRowMajor ? y * dim_size<1, ChunkShape>
                                 : y * Tile::kRows * dim_size<1, ChunkShape>;

        Iter iter(data_ + offset);
        return iter;
    }

    DEVICE auto to_tile() {
        Tile tile(data_);
        return tile;
    }

  private:
    typename Tile::DType* data_;
};

}  // namespace tiledcuda::cell
