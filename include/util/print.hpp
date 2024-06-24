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
    if (!thread0()) return;

    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            printf("%.1f, ", static_cast<float>(data[layout(i, j)]));
        }
        printf("\n");
    }
    printf("\n");
}

template <typename Layout>
DEVICE void print_tile(const cutlass::half_t* data, const Layout& layout) {
    if (!thread0()) return;

    const half* data_ = reinterpret_cast<const half*>(data);

    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            printf("%.0f, ", __half2float(data_[layout(i, j)]));
        }
        printf("\n");
    }
    printf("\n");
}

}  // namespace cell
}  // namespace tiledcuda
