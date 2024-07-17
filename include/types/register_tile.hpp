#pragma once

// FIXME(haruhi): rename this file to `regishter.hpp` after all the refactors
// are compeleted.

#include "types/base_tile.hpp"
#include "util/print.hpp"

namespace tiledcuda::cell {

namespace tl = tile_layout;

// FIXME: rename to RegTile after all the refactors are compeleted
template <typename Element, typename Layout_>
class RegisterTile {
  public:
    static_assert(std::is_same_v<Element, float> ||
                      std::is_same_v<Element, cutlass::half_t> ||
                      std::is_same_v<Element, __half>,
                  "Unsupported type.");

    using DType = BaseTile<Element>;
    using Layout = Layout_;

    static constexpr int kNumel = tl::get_numel<Layout>;
    static constexpr int kRows = tl::num_rows<Layout>;
    static constexpr int kCols = tl::num_cols<Layout>;

    DEVICE RegisterTile() : layout_(Layout{}) {
        memset((void*)data_, 0, sizeof(data_));
    }

    DEVICE DType* mutable_data() { return (DType*)data_; }

    DEVICE const DType* data() const { return (DType*)data_; }

    // for write access
    DEVICE DType& operator()(int x, int y) { return data_[layout_(x, y)]; }

    // for read access
    DEVICE const DType& operator()(int x, int y) const {
        return data_[layout_(x, y)];
    }

  private:
    DType data_[kNumel];
    Layout layout_;
};

}  // namespace tiledcuda::cell
