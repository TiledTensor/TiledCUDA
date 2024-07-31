#pragma once

#include "types/shared.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell {
namespace tl = tile_layout;

namespace detail {
/// @brief Helper for pretty printing a tile iterator's static shape-related
///        information. This printer works ONLY on the host.
struct TileIteratorPrettyPrinter {
    template <typename TileIterator>
    static HOST void print(std::ostream& out, const TileIterator& itr) {
        out << "numel = " << TileIterator::Tile::kNumel << ", ChunkShape["
            << dim_size<0, typename TileIterator::ChunkShape> << ", "
            << dim_size<1, typename TileIterator::ChunkShape> << "], sc0 = "
            << TileIterator::sc0 << ", sc1 = " << TileIterator::sc1;
    }
};
}  // namespace detail

struct Underscore {};                  // dummy type for underscore
static const __device__ Underscore _;  // for slicing

/// @brief `SharedTileIterator` chunks a shared memory tile into smaller tiles
///         and iterates over these smaller sub-tiles.
/// @tparam Tile_: The type of the large tile to chunk.
/// @tparam ChunkShape_: The shape of the smaller tiles into which the large
///                      tile is partitioned (chunk shape).
template <class Tile_, class ChunkShape_>
class TileIterator {
  public:
    using Tile = Tile_;
    using DType = Tile::DType;
    using ChunkShape = ChunkShape_;

    static_assert(Tile::kRows >= dim_size<0, ChunkShape>,
                  "Tile::kRows must be >= dim_size<0, ChunkShape>");
    static_assert(Tile::kCols >= dim_size<1, ChunkShape>,
                  "Tile::kCols must be >= dim_size<1, ChunkShape>");

    static constexpr int kStride0 = dim_size<0, ChunkShape>;
    static constexpr int kStride1 = dim_size<1, ChunkShape>;

    static constexpr int sc0 = Tile::kRows / kStride0;
    static constexpr int sc1 = Tile::kCols / kStride1;

    static constexpr bool kSwizzled = Tile::Layout::kSwizzled;

    HOST_DEVICE TileIterator() : data_(nullptr) {}

    DEVICE TileIterator(DType* data) : data_(data) {}

    DEVICE TileIterator(const DType* data) : data_(const_cast<DType*>(data)) {}

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

        using LayoutMaker =
            tl::LayoutMaker<kSwizzled, kStride0, kStride1, Tile::kRowStride,
                            Tile::kColStride, kB, kM, kS>;
        LayoutMaker maker;
        using TileLayout = decltype(maker());

        // using TileLayout = decltype(tl::make_tile_layout<kSwizzled>(
        //     kStride0, kStride1, Tile::kRowStride, Tile::kColStride, kB, kM,
        //     kS));

        using NewTile = SharedTile<DType, TileLayout>;

        int offset = Tile::kType == tl::Layout::kRowMajor
                         ? x * (kStride0 * Tile::kRowStride) + y * kStride1
                         : x * kStride0 + y * (Tile::kColStride * kStride1);

        NewTile tile(data_ + offset);

        return tile;
    }

    DEVICE auto operator()(int x, int y) {
        assert(data_);               // The iterator is not initialized.
        assert(x < sc0 && y < sc1);  // indices must be within the strip count.

        // using TileLayout = decltype(tl::make_tile_layout<kSwizzled>(
        //     kStride0, kStride1, Tile::kRowStride, Tile::kColStride, kB, kM,
        //     kS));

        using LayoutMaker =
            tl::LayoutMaker<kSwizzled, kStride0, kStride1, Tile::kRowStride,
                            Tile::kColStride, kB, kM, kS>;
        LayoutMaker maker;
        using TileLayout = decltype(maker());

        using NewTile = SharedTile<DType, TileLayout>;

        int offset = Tile::kType == tl::Layout::kRowMajor
                         ? x * (kStride0 * Tile::kCols) + y * kStride1
                         : x * kStride0 + y * (Tile::kRows * kStride1);
        NewTile tile(data_ + offset);

        return tile;
    }

    DEVICE auto operator()(int x, const Underscore& y) {
        assert(data_);    // The iterator is not initialized.
        assert(x < sc0);  // index must be within the strip count.

        // Updated the layout for sub-tiles accessed by the sliced iterator.
        // Note: Only the shape changes; the stride remains the same.
        using LayoutMaker =
            tl::LayoutMaker<kSwizzled, kStride0, Tile::kCols, Tile::kRowStride,
                            Tile::kColStride, kB, kM, kS>;
        LayoutMaker maker;
        using TileLayout = decltype(maker());

        // using TileLayout = decltype(tl::make_tile_layout<kSwizzled>(
        //     kStride0, Tile::kCols, Tile::kRowStride, Tile::kColStride, kB,
        //     kM, kS));

        using NewTile = SharedTile<DType, TileLayout>;
        using Iter = TileIterator<NewTile, ChunkShape>;
        static_assert(Iter::sc0 == 1);

        // advance pointer to the correct start position
        int offset = Tile::kType == tl::Layout::kRowMajor
                         ? x * (kStride0 * Tile::kCols)
                         : x * kStride0;

        Iter iter(data_ + offset);
        return iter;
    }

    DEVICE auto operator()(const Underscore& x, int y) {
        assert(data_);    // The iterator is not initialized.
        assert(y < sc1);  // index must be within the strip count.

        // Updated the layout for sub-tiles accessed by the sliced iterator.
        // Note: Only the shape changes; the stride remains the same.
        using LayoutMaker =
            tl::LayoutMaker<kSwizzled, Tile::kRows, kStride1, Tile::kRowStride,
                            Tile::kColStride, kB, kM, kS>;
        LayoutMaker maker;
        using TileLayout = decltype(maker());

        // using TileLayout = decltype(tl::make_tile_layout<kSwizzled>(
        //     Tile::kRows, kStride1, Tile::kRowStride, Tile::kColStride, kB,
        //     kM, kS));

        using NewTile = SharedTile<DType, TileLayout>;
        using Iter = TileIterator<NewTile, ChunkShape>;
        static_assert(Iter::sc1 == 1);

        // advance pointer to the correct start position
        int offset = Tile::kType == tl::Layout::kRowMajor
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
    static constexpr int kB = Tile::Layout::kB;
    static constexpr int kM = Tile::Layout::kM;
    static constexpr int kS = Tile::Layout::kS;
    DType* data_;
};

/// @brief Pretty printer for the static shape information of a TileIterator.
///        Note: This printer function works ONLY on the host.
template <typename TileShape, typename ChunkShape>
static HOST std::ostream& operator<<(
    std::ostream& out, const TileIterator<TileShape, ChunkShape>& itr) {
    detail::TileIteratorPrettyPrinter::print(out, itr);
    return out;
}

}  // namespace tiledcuda::cell
