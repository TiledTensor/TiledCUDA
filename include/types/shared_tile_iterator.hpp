#pragma once

#include "cell/traits/base.hpp"
#include "types/shared.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell {
namespace tl = tile_layout;

using namespace cute;

namespace detail {
/// @brief Helper for pretty printing a tile iterator's static shape-related
///        information. This printer works ONLY on the host.
struct STileIteratorPrettyPrinter {
    template <typename TileIterator>
    static HOST void print(std::ostream& out, const TileIterator& itr) {
        out << "numel = " << TileIterator::Tile::kNumel << ", ChunkShape["
            << dim_size<0, typename TileIterator::ChunkShape> << ", "
            << dim_size<1, typename TileIterator::ChunkShape> << "], sc0 = "
            << TileIterator::sc0 << ", sc1 = " << TileIterator::sc1;
    }
};
}  // namespace detail

/// @brief `SharedTileIterator` chunks a shared memory tile into smaller tiles
///         and iterates over these smaller sub-tiles.
/// @tparam Tile_: The type of the large tile to chunk.
/// @tparam ChunkShape_: The shape of the smaller tiles into which the large
///                      tile is partitioned (chunk shape).
template <class Tile_, class ChunkShape_>
class STileIterator {
  public:
    using Tile = Tile_;
    using DType = Tile::DType;
    using ChunkShape = ChunkShape_;
    using BaseShape = traits::BaseTileShape<DType>;

    static_assert(Tile::kRows >= dim_size<0, ChunkShape>,
                  "Tile::kRows must be >= dim_size<0, ChunkShape>");
    static_assert(Tile::kCols >= dim_size<1, ChunkShape>,
                  "Tile::kCols must be >= dim_size<1, ChunkShape>");

    static constexpr int kChunkRow = dim_size<0, ChunkShape>;
    static constexpr int kChunkCol = dim_size<1, ChunkShape>;

    static constexpr int sc0 = Tile::kRows / kChunkRow;
    static constexpr int sc1 = Tile::kCols / kChunkCol;

    HOST_DEVICE STileIterator() : data_(nullptr) {}

    DEVICE STileIterator(DType* data) : data_(data) {}

    DEVICE STileIterator(const DType* data) : data_(const_cast<DType*>(data)) {}

    // Since a Tile is considered to be at most a 2D array, the iterator
    // traverses over these two dimensions. The current rules are:
    // 1. If the index is a 2D integer, this access is considered to be a
    //    single tile, hence it returns a Tile.
    // 2. If any part of the index is an underscore, this access is
    //    considered to be a slice, naturally it returns a TileIterator.
    DEVICE auto operator()(int i) {
        assert(data_);  // The iterator is not initialized.
        static_assert(sc0 == 1 || sc1 == 1,
                      "A single index is supported only when the strip count "
                      "of one of the iterator's dimensions is 1.");

        int x = sc0 == 1 ? 0 : i;
        int y = sc0 == 1 ? i : 0;

        using TileLayout =
            decltype(tl::make_shared_tile_layout<kChunkRow, kChunkCol,
                                                 kTileRowStride, kTileColStride,
                                                 Tile::kType>());

        using NewTile = SharedTile<DType, TileLayout, Tile::kSwizzled>;

        int offset1 = x * (kChunkRow * Tile::kRowStride) +
                      y * kTilePerChunkCol * BaseShape::kNumel;
        int offset2 = x * kTilePerChunkRow * BaseShape::kNumel +
                      y * (Tile::kColStride * kChunkCol);
        int offset = Tile::kType == tl::Layout::kRowMajor ? offset1 : offset2;

        NewTile tile(data_ + offset);
        return tile;
    }

    DEVICE auto operator()(int x, int y) {
        assert(false && "Not implemented yet.");
        return 0;
    }

    DEVICE auto operator()(int x, const Underscore& y) {
        assert(false && "Not implemented yet.");
        return 0;
    }

    DEVICE auto operator()(const Underscore& x, int y) {
        assert(false && "Not implemented yet.");
        return 0;
    }

    DEVICE auto to_tile() {
        Tile tile(data_);
        return tile;
    }

  private:
    static constexpr int kTilePerRow = Tile::kRows / BaseShape::kRows;
    static constexpr int kTilePerCol = Tile::kCols / BaseShape::kCols;

    static constexpr int kTilePerChunkRow = kChunkRow / BaseShape::kRows;
    static constexpr int kTilePerChunkCol = kChunkCol / BaseShape::kCols;

    // The shared memory tile iterator creates a sub-tile that spans multiple
    // `BaseTile`s. The row and column strides are used to address a single
    // `BaseTile`. DO NOT modify these unless you fully understand how this
    // layout is used with the Shared to Register loader, as changes might
    // cause significant errors.
    static constexpr int kTileRowStride = Tile::kType == tl::Layout::kRowMajor
                                              ? kTilePerCol * BaseShape::kNumel
                                              : BaseShape::kNumel;

    static constexpr int kTileColStride = Tile::kType == tl::Layout::kRowMajor
                                              ? BaseShape::kNumel
                                              : kTilePerRow * BaseShape::kNumel;

    DType* data_;
};

/// @brief Pretty printer for the static shape information of a TileIterator.
///        Note: This printer function works ONLY on the host.
template <typename TileShape, typename ChunkShape>
static HOST std::ostream& operator<<(
    std::ostream& out, const STileIterator<TileShape, ChunkShape>& itr) {
    detail::STileIteratorPrettyPrinter::print(out, itr);
    return out;
}

}  // namespace tiledcuda::cell
