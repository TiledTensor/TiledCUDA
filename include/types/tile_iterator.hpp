#pragma once

#include "types/shared.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell {
namespace tl = tile_layout;

struct Underscore {};                  // dummy type for underscore
static const __device__ Underscore _;  // for slicing

template <class Tile, class ChunkShape>
struct TileIterator;

/// @brief `SharedTileIterator` chunks a shared memory tile into smaller tiles
///         and iterates over these smaller sub-tiles.
/// @tparam Tile_: The type of the large tile to chunk.
/// @tparam ChunkShape_: The shape of the smaller tiles into which the large
///                      tile is partitioned (chunk shape).
template <class Tile_, class ChunkShape_>
class TileIterator {
  public:
    using Tile = Tile_;
    using ChunkShape = ChunkShape_;

    static_assert(Tile::kRows >= dim_size<0, ChunkShape>,
                  "Tile::kRows must be >= dim_size<0, ChunkShape>");
    static_assert(Tile::kCols >= dim_size<1, ChunkShape>,
                  "Tile::kCols must be >= dim_size<1, ChunkShape>");

    static constexpr int kStride0 = dim_size<0, ChunkShape>;
    static constexpr int kStride1 = dim_size<1, ChunkShape>;

    static constexpr int sc0 = Tile::kRows / kStride0;
    static constexpr int sc1 = Tile::kCols / kStride1;

    DEVICE TileIterator(typename Tile::DType* data) : data_(data) {}

    // Since a Tile is considered to be at most a 2D array, the iterator
    // traverses over these two dimensions. The current rules are:
    // 1. If the index is a 2D integer, this access is considered to be a
    //    single tile, hence it returns a Tile.
    // 2. If any part of the index is an underscore, this access is
    //    considered to be a slice, naturally it returns a TileIterator.

    DEVICE auto operator()(int i) {
        static_assert(sc0 == 1 || sc1 == 1,
                      "A single index is supported only when the strip count "
                      "of one of the iterator's dimensions is 1.");

        int x = sc0 == 1 ? 0 : i;
        int y = sc0 == 1 ? i : 0;

        using TileLayout =
            decltype(tl::make_tile_layout<kStride0, kStride1, Tile::kRowStride,
                                          Tile::kColStride>());
        using NewTile = SharedTile<typename Tile::DType, TileLayout>;

        int offset = Tile::type == tl::Layout::RowMajor
                         ? x * (kStride0 * Tile::kRowStride) + y * kStride1
                         : x * kStride0 + y * (Tile::kColStride * kStride1);

        NewTile tile(data_ + offset);

        return tile;
    }

    DEVICE auto operator()(int x, int y) {
        // The indices must be within the strip count.
        assert(x < sc0 && y < sc1);

        using TileLayout =
            decltype(tl::make_tile_layout<kStride0, kStride1, Tile::kRowStride,
                                          Tile::kColStride>());
        using NewTile = SharedTile<typename Tile::DType, TileLayout>;

        int offset = Tile::type == tl::Layout::RowMajor
                         ? x * (kStride0 * Tile::kCols) + y * kStride1
                         : x * kStride0 + y * (Tile::kRows * kStride1);
        NewTile tile(data_ + offset);

        return tile;
    }

    DEVICE auto operator()(int x, const Underscore& y) {
        assert(x < sc0);  // The index must be within the strip count.

        // Updated the layout for sub-tiles accessed by the sliced iterator.
        // Note: Only the shape changes; the stride remains the same.
        using TileLayout = decltype(tl::make_tile_layout<kStride0, Tile::kCols,
                                                         Tile::kRowStride,
                                                         Tile::kColStride>());

        using NewTile = SharedTile<typename Tile::DType, TileLayout>;
        using Iter = TileIterator<NewTile, ChunkShape>;
        static_assert(Iter::sc0 == 1);

        // advance pointer to the correct start position
        int offset = Tile::type == tl::Layout::RowMajor
                         ? x * (kStride0 * Tile::kCols)
                         : x * kStride0;

        Iter iter(data_ + offset);
        return iter;
    }

    DEVICE auto operator()(const Underscore& x, int y) {
        assert(y < sc1);  // The index must be within the strip count.

        // Updated the layout for sub-tiles accessed by the sliced iterator.
        // Note: Only the shape changes; the stride remains the same.
        using TileLayout = decltype(tl::make_tile_layout<Tile::kRows, kStride1,
                                                         Tile::kRowStride,
                                                         Tile::kColStride>());

        using NewTile = SharedTile<typename Tile::DType, TileLayout>;
        using Iter = TileIterator<NewTile, ChunkShape>;
        static_assert(Iter::sc1 == 1);

        // advance pointer to the correct start position
        int offset = Tile::type == tl::Layout::RowMajor
                         ? y * kStride1
                         : y * (Tile::kRows * kStride1);

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
