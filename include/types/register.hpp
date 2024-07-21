#pragma once

#include "cuda_utils.hpp"
#include "types/layout.hpp"
#include "util/print.hpp"

namespace tiledcuda::cell {
namespace tl = tile_layout;

namespace detail {

namespace {
template <typename DType>
constexpr int get_rows = DType::kRows;

template <>
constexpr int get_rows<float> = 1;

template <>
constexpr int get_rows<__half> = 1;

template <>
constexpr int get_rows<cutlass::half_t> = 1;

template <typename DType>
constexpr int get_cols = DType::kCols;

template <>
constexpr int get_cols<float> = 1;

template <>
constexpr int get_cols<__half> = 1;

template <>
constexpr int get_cols<cutlass::half_t> = 1;
}  // namespace

/// @brief Helper for pretty printing a register tile's static shape
/// information. This printer works ONLY on the host.
struct RegTilePrettyPrinter {
    template <typename Tile>
    static HOST void print(std::ostream& out, const Tile& tile) {
        out << layout_type_to_str(Tile::kType) << "["
            << Tile::kRows * get_rows<typename Tile::DType> << ", "
            << Tile::kCols * get_cols<typename Tile::DType> << "]";
    }
};

}  // namespace detail

template <typename Element_, typename Layout_>
class RegTile {
  public:
    using DType = Element_;
    using Layout = Layout_;

    static constexpr int kNumel = tl::get_numel<Layout>;
    static constexpr int kRows = tl::num_rows<Layout>;
    static constexpr int kCols = tl::num_cols<Layout>;

    static constexpr tl::Layout kType = tl::layout_type<Layout>;

    DEVICE RegTile() : layout_(Layout{}) {
        memset((void*)data_, 0, sizeof(data_));
    }

    DEVICE DType* mutable_data() { return (DType*)data_; }

    DEVICE const DType* data() const { return (DType*)data_; }

    DEVICE const Layout& layout() const { return layout_; }

    // for write access
    DEVICE DType& operator()(int x, int y) { return data_[layout_(x, y)]; }

    // for read access
    DEVICE const DType& operator()(int x, int y) const {
        return data_[layout_(x, y)];
    }

    DEVICE void dump_value() {
        print_tile(static_cast<DType*>(data_), layout_);
    }

  private:
    DType data_[kNumel];
    Layout layout_;
};

template <typename Element>
using BaseTileRowMajor = RegTile<Element, tl::RowMajor<2, 4>>;

template <typename Element>
using BaseTileColMajor = RegTile<Element, tl::ColMajor<4, 2>>;

/// @brief Pretty printer for the static shape information of a register tile.
///        Note: This printer function works ONLY on the host. The current
///        implementation provides a flattened layout and only displays the
///        outer name of the tile layout.
/// @tparam T: element type, which must be a `RegTile` rather than a basic
///            element type like float
/// @tparam Layout: tile layout
template <typename T, typename Layout>
static HOST std::ostream& operator<<(std::ostream& out,
                                     const RegTile<T, Layout>& tile) {
    detail::RegTilePrettyPrinter::print(out, tile);
    return out;
}

}  // namespace tiledcuda::cell
