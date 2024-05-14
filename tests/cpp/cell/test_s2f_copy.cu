#include "cell/copy/mod.hpp"
#include "cell/sync.hpp"
#include "cell/traits/copy.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda {

using namespace cute;
namespace traits = cell::traits;
namespace tl = tile_layout;

namespace {
template <typename Element>
__device__ void init_data(Element* data, int64_t numel) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < numel; ++i) {
            data[i] = static_cast<Element>(i);
        }
    }
}

template <typename Element>
__device__ void debug_print(Element* data, int rows, int cols) {
    if (threadIdx.x == 0) {
        printf("\ntile shape = [%d, %d]\n", rows, cols);
        for (int i = 0; i < rows; ++i) {
            printf("%d: ", i);

            for (int j = 0; j < cols - 1; ++j) {
                printf("%.0f, ", static_cast<float>(data[i * cols + j]));
            }
            printf("%.0f\n", static_cast<float>(data[(i + 1) * cols - 1]));
        }
        printf("\n");
    }
}

template <typename Element, typename SrcLayout, typename DstLayout>
__global__ void copy_s2r(int rows, int cols) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init_data(buf, rows * cols);
    // debug_print(buf, rows, cols);

    SharedTile<Element, SrcLayout> s_tile(buf);
    RegTile<Element, DstLayout> r_tile;

    for (int k = 0; k < 2; ++k) {
        cell::copy::copy_2d_tile_s2r(s_tile[k * cols], r_tile);

        // cell::copy::copy_2d_tile_s2r(s_tile, r_tile);
    }
}
}  // namespace

namespace testing {
TEST(TestShm2Rf, copy_2d_tile_s2r) {
    using Element = cutlass::half_t;

    const int kRows = 16;
    const int kCols = 32;

    // swizzled row major layout for shared memory
    using Swizzled = tl::SwizzledRowMajor<Element, kRows, kCols, 0>;
    using SrcLayout = typename Swizzled::SmemLayout;

    using DstLayout = tl::RowMajor<kRows, kCols>;
    using ThreadLayout = tl::RowMajor<16, 2>;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(32, 1, 1);

    int shm_size = kRows * kCols * sizeof(Element);

    // using S2RCopyTraits =
    //     traits::S2R2DCopyTraits<Element, SrcLayout, DstLayout, ThreadLayout>;
    copy_s2r<Element, SrcLayout, DstLayout>
        <<<dim_grid, dim_block, shm_size>>>(kRows, kCols);
    cudaDeviceSynchronize();
}

}  // namespace testing
}  // namespace tiledcuda
