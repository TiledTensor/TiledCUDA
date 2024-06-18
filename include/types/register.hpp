#pragma once

#include "cell/traits/base.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell {

namespace tl = tile_layout;

namespace detail {

// for debugging purpose
template <typename Element>
DEVICE void dump_value(const Element* data, int numel);

template <>
DEVICE void dump_value<uint32_t>(const uint32_t* data, int numel) {
    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    const half* reg = reinterpret_cast<const half*>(data);

    for (int i = 0; i < numel / 8; i += 8) {
        printf("[tid = %d]: %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f\n",
               threadIdx.x, __half2float(reg[0]), __half2float(reg[1]),
               __half2float(reg[2]), __half2float(reg[3]), __half2float(reg[4]),
               __half2float(reg[5]), __half2float(reg[6]),
               __half2float(reg[7]));
    }

    if (numel % 8) {
        printf("[tid = %d]: ", threadIdx.x);
        for (int i = numel / 8 * 8; i < numel - 1; ++i)
            printf("%.0f, ", __half2float(reg[i]));
        printf("%.0f\n\n", __half2float(reg[numel - 1]));
    }
}
}  // namespace detail

template <class Element_, class TileLayout_>
class Reg {
  public:
    using DType = Element_;
    using TileLayout = TileLayout_;

    static constexpr int kRows = tl::num_rows<TileLayout>;
    static constexpr int kCols = tl::num_cols<TileLayout>;

    DEVICE Reg() { memset((void*)data_, 0, sizeof(data_)); }

    DEVICE DType* operator[](int2 idx) { return &data_[idx.x][idx.y]; }

    DEVICE const DType* operator[](int2 idx) const {
        return &data_[idx.x][idx.y];
    }

    DEVICE DType& operator[](int idx) {
        DType* ptr = (DType*)data_;
        return ptr[idx];
    }

    DEVICE const DType& operator[](int idx) const {
        DType* ptr = (DType*)data_;
        return ptr[idx];
    }

    DEVICE DType* mutable_data() { return (DType*)data_; }

    DEVICE const DType* data() const { return (DType*)data_; }

    template <typename T>
    DEVICE static void dump_value(const T* data, int numel) {
        detail::dump_value<T>(data, numel);
    };

  private:
    // register tile data is implemented as an 2D array of size kRows x kCols
    DType data_[kRows][kCols];
};

}  // namespace tiledcuda::cell
