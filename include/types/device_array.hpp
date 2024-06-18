#pragma once

#include "types/layout.hpp"

namespace tiledcuda::cell {

namespace tl = tile_layout;

/// @brief: A fixed-length small 1D array on the device.
/// usage: store a set of data pointers on device.
template <typename Element_, const int N_>
class DevArray {
  public:
    using Element = Element_;
    constexpr static int kSize = N_;

    DEVICE DevArray() { memset((void*)data_, 0, sizeof(data_)); }

    DEVICE Element& operator[](int idx) { return data_[idx]; }

  private:
    Element data_[kSize];
};

template <typename DType, typename Layout>
struct PrintTile {
    DEVICE void operator()(const DType* data, const Layout& layout);
};

template <typename Layout>
struct PrintTile<cutlass::half_t, Layout> {
    DEVICE void operator()(const cutlass::half_t* data, const Layout& layout) {
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
};

}  // namespace tiledcuda::cell
