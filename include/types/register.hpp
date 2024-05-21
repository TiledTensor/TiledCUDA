#pragma once

#include "cell/traits/base.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell {

/*
 * @brief The register tile is comprised of a 2-dimensional array of data
 *        elements.
 * @tparam Element_: the data type of the elements.
 * @tparam TemporalExec_: the atomic memory access instruction is executed in a
 *         temporal fashion along two dimensions. The first element of
 *         TemporalExec_ is the number of stripes along the outer dimension,
 *         and the second element is the number of stripes along the inner
 *         dimension.
 * @tparam ElemTileLayout_: the data tile layout for executing a single atomic
 *         memory access instruction at a time.
 */
template <typename Element, typename TemporalExec_, typename ElemTileLayout_,
          typename Base = traits::TraitsBase<Element>>
class RegTile : public Base {
  public:
    using DType = Element;
    using TemporalExec = TemporalExec_;
    using ElemTileLayout = ElemTileLayout_;

    constexpr static int kRows =
        dim_size<0, ElemTileLayout> * dim_size<0, TemporalExec>;
    constexpr static int kCols =
        dim_size<1, ElemTileLayout> * dim_size<1, TemporalExec>;

    // how many times the atomic memory access instruction is executed
    constexpr static int kExecCount = get_numel<TemporalExec>;

    constexpr static int kElemInsNumel = get_numel<ElemTileLayout>;
    constexpr static int kNumel = kRows * kCols;

    DEVICE RegTile() {
        // memset((void*)data_, 0, sizeof(data_));
        int count = 0;
        for (int i = 0; i < kRows; i++)
            for (int j = 0; j < kCols; j++) {
                data_[i][j] = static_cast<DType>(++count);
            }
    }

    DEVICE DType* operator[](int2 idx) { return &data_[idx.x][idx.y]; }

    DEVICE const DType* operator[](int2 idx) const {
        return &data_[idx.x][idx.y];
    }

    DEVICE DType* mutable_data() { return (DType*)data_; }

    DEVICE const DType* data() const { return (DType*)data_; }

  private:
    // register tile data is implemented as an 2D array of size kRows x kCols
    DType data_[kRows][kCols];
};

}  // namespace tiledcuda::cell
