#pragma once

#include "config.hpp"
#include "types/layout.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell {

namespace tl = tile_layout;

template <typename Element_, typename TemporalExec_, typename DataLayout_>
class RegTile {
  public:
    using DType = Element_;
    using TemporalExec = TemporalExec_;
    using DataLayout = DataLayout_;

    constexpr static int kRows =
        cell::dim_size<0, DataLayout> * cell::dim_size<0, TemporalExec>;
    constexpr static int kCols =
        cell::dim_size<1, DataLayout> * cell::dim_size<1, TemporalExec>;

    DEVICE DType* operator[](int2 idx) { return &data_[idx.x][idx.y]; }

    DEVICE const DType* operator[](int2 idx) const {
        return &data_[idx.x][idx.y];
    }

    DEVICE DType* mutable_data() { return (DType*)data_; }

    DEVICE const DType* data() const { return (DType*)data_; }

  private:
    DType data_[kRows][kCols];  // register tile data is implemented as an 2D
                                // array of size kRows x kCols
};

}  // namespace tiledcuda::cell
