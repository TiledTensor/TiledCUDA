#pragma once

#include "config.hpp"
#include "types/layout.hpp"

namespace tiledcuda {

namespace tl = tile_layout;

namespace {
// ======================  The Register tile =================================
// @brief: a fixed length array that stores on a thread's local register file,
// with length being 16 and element type being the half precision floating
// numbers.
typedef struct reg_16 {
    uint32_t tile[2];  // 2 128-bit registers, can store 16 halfs
} reg_16;
// ============================================================================
}  // namespace

template <typename Element_, typename Layout_>
class RegTile {
  public:
    using Element = Element_;
    using Layout = Layout_;  // to keep the behavior consistent with SharedTile

    constexpr static int kRows = tl::num_rows<Layout_>;
    constexpr static int kCols = tl::num_cols<Layout_>;

    DEVICE Element* operator[](int2 idx) { return &data_[idx.x][idx.y]; }

    DEVICE const Element* operator[](int2 idx) const {
        return &data_[idx.x][idx.y];
    }

    DEVICE Element* mutable_data() { return (Element*)data_; }

    DEVICE const Element* data() const { return (Element*)data_; }

  private:
    Element data_[kRows][kCols];  // register tile data
};

}  // namespace tiledcuda
