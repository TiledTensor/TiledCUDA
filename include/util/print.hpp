#pragma once

#include "types/layout.hpp"
#include "util/debug.hpp"

namespace tiledcuda::cell {

namespace tl = tile_layout;

/// @brief Print a tile of single-precision floating point numbers. NOTE: when
// use print in the device function, do add (if(thread0())) to avoid printing
// multiple times by multiple threads. usage:
// if(thread0()) {
//   print_tile(data, layout);
// }
template <typename Layout>
DEVICE void print_tile(const float* data, const Layout& layout) {
    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            printf("%.1f, ", data[layout(i, j)]);
        }
        printf("\n");
    }
}

/// @brief Print a tile of half-precision floating point numbers.
template <typename Layout>
DEVICE void print_tile(const cutlass::half_t* data, const Layout& layout) {
    const half* data_ = reinterpret_cast<const half*>(data);

    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            printf("%.1f, ", __half2float(data_[layout(i, j)]));
        }
        printf("\n");
    }
}

/// @brief Print a tile of half-precision floating point numbers.
template <typename Layout>
DEVICE void print_tile(const __half* data, const Layout& layout) {
    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            printf("%.1f, ", __half2float(data[layout(i, j)]));
        }
        printf("\n");
    }
}

/// @brief Print a register tile. Since register tile is a nested array-like
///        structure. printing resigter tile hits this function.
template <typename DType, typename Layout>
DEVICE void print_tile(const DType* data, const Layout& layout) {
    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            auto tile = data[layout(i, j)];
            print_tile(tile.data(), tile.layout());
        }
    }
}

}  // namespace tiledcuda::cell
