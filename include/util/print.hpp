#pragma once

#include "types/layout.hpp"
#include "util/debug.hpp"

namespace tiledcuda {
namespace cell {

namespace tl = tile_layout;

// TODO: a naive implementation for printing a tile.
// improve it to be pretty printing and be compatible with more data types.
template <typename DType, typename Layout>
DEVICE void print_tile(const DType* data, const Layout& layout) {
    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            printf("%.2f, ", static_cast<float>(data[layout(i, j)]));
        }
        printf("\n");
    }
    printf("\n");
}

// @brief Print a tile of half-precision floating point numbers. NOTE: when use
// print in the device function, do add (if(thread0())) to avoid printing
// multiple times by multiple threads. usage:
// if(thread0()) {
//   print_tile(data, layout);
// }
template <typename Layout>
DEVICE void print_tile(const cutlass::half_t* data, const Layout& layout) {
    const half* data_ = reinterpret_cast<const half*>(data);

    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            printf("%.2f, ", __half2float(data_[layout(i, j)]));
        }
        printf("\n");
    }
    printf("\n");
}

}  // namespace cell
}  // namespace tiledcuda
