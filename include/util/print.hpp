#pragma once

#include "types/layout.hpp"

namespace tiledcuda {
namespace cell {

namespace tl = tile_layout;

template <typename DType, typename Layout>
DEVICE void print_tile(const DType* data, const Layout& layout) {
    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    const half* data_ = reinterpret_cast<const half*>(data);

    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            printf("%.1f, ", __half2float(data_[layout(i, j)]));
        }
        printf("\n");
    }
    printf("\n");
}

template <typename Layout>
DEVICE void print_tile(const cutlass::half_t* data, const Layout& layout) {
    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    const half* data_ = reinterpret_cast<const half*>(data);

    for (int i = 0; i < tl::num_rows<Layout>; ++i) {
        for (int j = 0; j < tl::num_cols<Layout>; ++j) {
            printf("%.1f, ", __half2float(data_[layout(i, j)]));
        }
        printf("\n");
    }
    printf("\n");
}

}  // namespace cell
}  // namespace tiledcuda
