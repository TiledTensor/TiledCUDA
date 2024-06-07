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

    // FIXME: The code below that determines whether the register tile is
    // created by ldmatrix or wmma is quite tricky. These two instructions leads
    // to different shared memory access patterns. It's important to fix this
    // tricky implementations to ensure that the code is more readable and
    // easier to work with.
    constexpr static bool kIsWmmaTile =
        tl::num_rows<ElemTileLayout> == 2 && tl::num_cols<ElemTileLayout> == 4;

    // how many 128-bit are used per access
    constexpr static int kRegCountPerAccess =
        Base::kAccessInBits / 8 / sizeof(uint32_t);

    constexpr static int kRows =
        tl::num_rows<ElemTileLayout> * dim_size<0, TemporalExec>;
    constexpr static int kCols =
        tl::num_cols<ElemTileLayout> * dim_size<1, TemporalExec>;

    // how many times the atomic memory access instruction is executed
    constexpr static int kExecCount = get_numel<TemporalExec>;

    constexpr static int kElemInsNumel = tl::get_numel<ElemTileLayout>;
    constexpr static int kNumel = kRows * kCols;

    DEVICE RegTile() { memset((void*)data_, 0, sizeof(data_)); }

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
